"""
t-SNE Visualization: Base vs ContrastSkill Embeddings
======================================================
Generates a side-by-side t-SNE plot showing how contrastive
intermediate training improves cluster separation for skill
competencies.

Usage:
    python tsne_visualization.py \
        --model_type joberta \
        --model_version jjzha/jobberta-base \
        --contrastive_weights /path/to/model_contrastive_stage.bin \
        --data_dir /path/to/Pre-training/ \
        --output tsne_comparison.png \
        --n_competencies 12 \
        --samples_per_comp 40
"""

import argparse
import json
import random
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load_model(model_version, device):
    """Load a fresh base model."""
    config = AutoConfig.from_pretrained(model_version)
    model = AutoModel.from_pretrained(model_version, config=config).to(device)
    model.eval()
    return model


def load_contrastive_model(model_version, contrastive_weights_path, device):
    """Load the base model and replace weights with contrastive-trained ones."""
    config = AutoConfig.from_pretrained(model_version)
    model = AutoModel.from_pretrained(model_version, config=config).to(device)
    
    state_dict = torch.load(contrastive_weights_path, map_location=device)
    # Replace layer names to match base encoder
    updated_state_dict = {
        k.replace('base_encoder.', ''): v 
        for k, v in state_dict.items() 
        if k.startswith('base_encoder.')
    }
    model.load_state_dict(updated_state_dict, strict=False)
    model.eval()
    return model


def get_weighted_embedding(model, tokenizer, sentence, device, 
                           relevant_tokens=None, weight_relevant=1.0):
    """
    Get sentence embedding using the same weighted pooling as ContrastSkill.
    If relevant_tokens provided, weight those tokens at weight_relevant.
    Otherwise, use mean pooling over all tokens.
    """
    inputs = tokenizer(
        sentence, 
        return_tensors='pt', 
        truncation=True, 
        max_length=128, 
        padding=True,
        return_special_tokens_mask=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**{k: v for k, v in inputs.items() 
                          if k in ['input_ids', 'attention_mask']})
        hidden_states = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)
    
    # Use attention mask to exclude padding
    attention_mask = inputs['attention_mask'][0].float()
    special_tokens_mask = inputs['special_tokens_mask'][0].float()
    
    # Simple mean pooling excluding special tokens and padding
    mask = attention_mask * (1 - special_tokens_mask)
    
    if mask.sum() == 0:
        mask = attention_mask  # fallback
    
    embedding = (hidden_states * mask.unsqueeze(-1)).sum(dim=0) / mask.sum()
    
    return embedding.cpu().numpy()


def prepare_data(data_dir, n_competencies=12, samples_per_comp=40, seed=42):
    random.seed(seed)
    
    pos_path = f"{data_dir}/selected_positives.json"
    with open(pos_path, 'r') as f:
        positives = json.load(f)
    
    # Group sentences by competency
    comp_to_sentences = {}
    for item in positives:
        comp = item.get('competence')
        tokens = item.get('Tokens')
        
        if comp is None or tokens is None:
            continue
        
        sent = ' '.join(tokens)
        
        if comp not in comp_to_sentences:
            comp_to_sentences[comp] = []
        comp_to_sentences[comp].append(sent)
    
    print(f"Found {len(comp_to_sentences)} unique competencies")
    
    # Filter to competencies with enough examples
    eligible = {k: v for k, v in comp_to_sentences.items() 
                if len(v) >= samples_per_comp}
    
    if len(eligible) < n_competencies:
        print(f"Only {len(eligible)} competencies have >= {samples_per_comp} examples, adjusting...")
        samples_per_comp = 20
        eligible = {k: v for k, v in comp_to_sentences.items() 
                    if len(v) >= samples_per_comp}
    
    # Select top n competencies by count
    selected_comps = sorted(eligible.keys(), 
                           key=lambda k: len(eligible[k]), 
                           reverse=True)[:n_competencies]
    
    data = []
    for comp in selected_comps:
        sampled = random.sample(eligible[comp], 
                               min(samples_per_comp, len(eligible[comp])))
        for sent in sampled:
            data.append({'competence': comp, 'sentence': sent})
    
    print(f"Selected {len(selected_comps)} competencies, {len(data)} total sentences")
    for comp in selected_comps:
        print(f"  {comp}: {len(eligible[comp])} available")
    
    return data, selected_comps


def extract_embeddings(model, tokenizer, data, device):
    """Extract embeddings for all sentences."""
    embeddings = []
    for item in data:
        emb = get_weighted_embedding(model, tokenizer, item['sentence'], device)
        embeddings.append(emb)
    return np.array(embeddings)


def plot_tsne(embeddings_base, embeddings_contrast, labels, comp_names, output_path):
    """Create two separate t-SNE plots."""
    
    perplexity = min(30, len(labels) - 1)
    
    # Compute t-SNE for both
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    coords_base = tsne.fit_transform(embeddings_base)
    coords_contrast = tsne.fit_transform(embeddings_contrast)
    
    # Professional color palette (12 distinct colors)
    palette = [
        '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
        '#dcbeff', '#9A6324'
    ]
    cmap = mcolors.ListedColormap(palette)
    
    # Create figure with 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    
    def plot_data(ax, coords, title):
        for i, comp in enumerate(comp_names):
            mask = [j for j, l in enumerate(labels) if l == comp]
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=palette[i % len(palette)], alpha=0.7, s=40,
                       edgecolors='none', zorder=2)
        
        ax.set_title(title, fontsize=25, pad=15)
        ax.set_facecolor('#fdfdfd')
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('#333333')

    # Plot both datasets
    plot_data(ax1, coords_base, "JobBERTa (base)")
    plot_data(ax2, coords_contrast, "JobBERTa (contrast)")

    # Add a single shared colorbar on the right
    norm = mcolors.BoundaryNorm(np.arange(len(palette) + 1), cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    fig.subplots_adjust(right=0.92, wspace=0.05)
    
    # Adjust layout to make room for colorbar
    # 
    # cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7]) # [left, bottom, width, height]
    # cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.arange(len(palette)) + 0.5)
    # cbar.ax.set_yticklabels([]) # Purely visual color representation
    # cbar.set_label('Top 12 Skills', rotation=270, labelpad=15, fontsize=10)
    
    # Save outputs
    base_name = output_path.rsplit('.', 1)[0]
    plt.savefig(f"{base_name}_combined.pdf", bbox_inches='tight')
    plt.savefig(f"{base_name}_combined.png", dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to {base_name}_combined.pdf")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='t-SNE visualization for ContrastSkill')
    parser.add_argument('--model_type', type=str, default='joberta')
    parser.add_argument('--model_version', type=str, default='jjzha/jobberta-base')
    parser.add_argument('--contrastive_weights', type=str, required=True,
                       help='Path to model_contrastive_stage.bin')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to Pre-training data directory')
    parser.add_argument('--output', type=str, default='tsne_comparison.png')
    parser.add_argument('--n_competencies', type=int, default=12)
    parser.add_argument('--samples_per_comp', type=int, default=40)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_version, add_prefix_space=True, use_fast=True
    )
    
    # Prepare data
    print("Preparing data...")
    data, comp_names = prepare_data(
        args.data_dir, args.n_competencies, args.samples_per_comp
    )
    labels = [item['competence'] for item in data]
    
    # Load base model and extract embeddings
    print("\nExtracting base model embeddings...")
    base_model = load_model(args.model_version, device)
    embeddings_base = extract_embeddings(base_model, tokenizer, data, device)
    del base_model
    torch.cuda.empty_cache()
    
    # Load contrastive model and extract embeddings
    print("Extracting ContrastSkill embeddings...")
    contrast_model = load_contrastive_model(
        args.model_version, args.contrastive_weights, device
    )
    embeddings_contrast = extract_embeddings(contrast_model, tokenizer, data, device)
    del contrast_model
    torch.cuda.empty_cache()
    
    # Plot
    print("Generating t-SNE visualization...")
    plot_tsne(embeddings_base, embeddings_contrast, labels, comp_names, args.output)


if __name__ == '__main__':
    main()
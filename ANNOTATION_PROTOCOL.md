# Evaluation Protocol for Skill Matching Rules (Stage 1)

This document describes the evaluation process used to assess the quality of the custom-designed competence matching rules used in Stage 1 of ContrastSkill.

## Overview

To quantify rule quality, we randomly sampled 1,691 job adverts (approximately 25,000 sentences), representing 10% of the 16,900 adverts harvested through the Lightcast API. These sentences are disjoint from every training, development, and test set used in the ContrastSkill pipeline. All identifiers were removed in compliance with Lightcast's Terms of Service.

## Recruitment and Compensation

Three graduate students with data backgrounds were selected from a pool of five candidates. Selection was based on a 10-sentence test exercise assessing skill extraction accuracy. Evaluators were remunerated according to standard University Pay Grades in Scotland.

## Training

Following the initial test exercise, evaluators were provided with a set of guidelines and requested to complete a training exercise consisting of 50 sentence examples, following the format identical to the final exercise. This was done to ensure participants were aware of the nature of the task and to address common mistakes prior to the actual evaluation. The samples were evaluated and feedback was discussed during a one-hour online training session with all participants present.

## Annotation Interface

Evaluation was conducted in Google Sheets, using the layout specified in the guidelines and used in the training exercise. Participants were asked to identify any incorrectly retrieved skills or knowledge, as well as provide a list of missing items if there were any. During the study, evaluators had access to a designated Google Chat workspace where they could discuss problematic cases. This workspace allowed for dynamic communication between the participants and the researcher.

## Inter-Annotator Agreement

To enable inter-annotator evaluation, we split the corpus into three separate evaluation sets (7,700 sentences each) and added a common pool of 2,000 sentences to every shard. Because annotators can add or discard skills, they produce variable-size sets rather than labels on a fixed grid. Token-aligned metrics such as Cohen's kappa are therefore less informative. We report agreement using the Sorensen-Dice coefficient, an overlap-based measure recommended for set-based annotations. Mean pairwise Dice across the shared pool is 88 +/- 7%, indicating high consistency between evaluators.

## Quality Control

Evaluation samples were inspected weekly, with random checks for common errors. Participants received written feedback on each batch, followed by a 30-minute weekly meeting to discuss issues and clarify concerns. This ensured that a consistent quality standard was maintained and gave evaluators engagement opportunities (e.g., discussing edge cases with colleagues).

## Duration

The entire evaluation study was conducted over a period of 12 weeks, with each participant completing their 7,700 samples in addition to the 2,000 shared examples.

## Results

| Metric    | Score  |
|-----------|--------|
| Precision | 91.1%  |
| Recall    | 94.3%  |
| F1        | 92.6%  |

The main residual error class is abstract wording, where sentences describe a duty rather than explicitly name the skill. For instance, "Manage the process and information of updates to price lists and notifications to relevant parties" implicitly requires "data management skill", but no canonical competence term appears, so the rule set misses it.

#Set the main parameters (refer to main.py for in-depth on all parameters)
DATA_PATH="./ContrastSkill/Data"
MODEL_PATH="./ContrastSkill/Models/Model_SKILLSPAN"
MODEL_TYPE="joberta" #(roberta, bert)
MODEL_VERSION="jjzha/jobberta-base"  #(FacebookAI/roberta-base, bert-base-uncased)
SUPERVISED_LEARNING_RATE=5e-5
DATASET="SkillSpan" #(Green, Sayfullina)
SUPERVISED_EPOCHS=19
CROSS_DATASET=0 #(0 for in dataset setting, 1 for SkillSpan on Green, 2 for Green on SkillSpan, remember to adjust the DATASET accordingly)
MODE=2
SEED=21

# Run the pipeline
python main.py \
    --data_directory $DATA_PATH \
    --model_directory $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --model_version $MODEL_VERSION \
    --supervised_dataset $DATASET \
    --supervised_learning_rate $SUPERVISED_LEARNING_RATE \
    --supervised_epochs $SUPERVISED_EPOCHS \
    --mode $MODE \
    --seed $SEED \
    --cross_dataset $CROSS_DATASET \
    --supervised_train \
    # --supervised_raw \
    # --contrastive_train \
    # --prepare_data \
    
    
    
    
    
    


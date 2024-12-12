#Set the main parameters
DATA_PATH="./ContrastSkill/Data"
MODEL_PATH="./ContrastSkill/Models/Model_SKILLSPAN_dev"
MODEL_TYPE="joberta"
MODEL_VERSION="jjzha/jobberta-base" #
SUPERVISED_LEARNING_RATE=5e-5
DATASET="SkillSpan" #(Green, SkillSpan, Sayfullina)
SUPERVISED_EPOCHS=20
PATIENCE=5
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
    --patience $PATIENCE \
    --supervised_train \
    --dev_experiments \
    # --supervised_raw \
    
    



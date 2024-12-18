
# Load legacy data


config = DataConfig(
    data_path='data/Categorized_Fixation_Data_1_18.csv',
    approach_num=6,
    normalize=True,
    per_slice_target=True,
    participant_id=1
)


df = load_eye_tracking_data(
    data_path=config.data_path,
    approach_num=config.approach_num,
    participant_id=config.participant_id,
    data_format="legacy"
)
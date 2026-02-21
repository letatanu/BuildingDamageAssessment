data_root = "/data/converted_mmflood"
output_dir = "/working/runs/mask2former_mmflood"
model_name = "facebook/mask2former-swin-large-cityscapes-semantic"
epochs = 300
train_split = "train"
val_split = "val"
test_split = "test"
batch_size = 4
fp16=True
ignore_index = 255
num_classes = 2

DATASET_NAME = "mmflood_mask2former"
DATASET_KWARGS = {
    "root": data_root,
    "image_size": 512,
    "augment": True,
    "ignore_index": ignore_index,
    "num_classes": num_classes
}
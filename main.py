import torch

from data.data_loader import VisionDataLoader
import config_ as st
from model import Model
from train_evaluate import TrainAndEvaluate
from visualization import Visualization

# if cuda is not available we need to switch to cpu
st.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Définir le type d'attention que vous voulez utiliser
attention_type = 'fsattention'  # ou 'scaled_dot_product', 'multihead', etc.

print("============ Load Data ========")
# Load data
visionDataLoader = VisionDataLoader()
data_loader_1, data_loader_2, data_loader_3 = visionDataLoader.get_hmdb51_data_loaders(st.DATA_PATH, st.TEST_SPLIT_PATH)

# Construct the model
if(st.MODEL_PATH == ""):
    print("============ Construct The Model ========")
    model = Model(st.NUM_CLASSES, attention_type=attention_type).to(st.DEVICE)
else:
    print("============ Load The Model ========")
    model = torch.load(st.MODEL_PATH).to(st.DEVICE)

# Prepare the training, if a model path is set so the model has already been trained; else we do the training
trainAndEvaluate = TrainAndEvaluate(model=model)
if(st.MODEL_PATH == ""):
    print("============ Train ========")
    loss_list, training_accuracy_list = trainAndEvaluate.train(data_loader_1, st.LEARNING_RATE, st.WEIGHT_DECAY,
                                                               intermediate_result_step=20, print_epoch_result_step=1)
    
# -----------------------------------------------
# Prepare test
st.IS_TRAIN = False
print("============ Load Test Data ========")
test_data_loader_1, test_data_loader_2, test_data_loader_3 = visionDataLoader.get_hmdb51_data_loaders(st.DATA_PATH, st.TEST_SPLIT_PATH)

# Perform tests
print("============ Perform Test ========")
trainAndEvaluate.test(test_data_loader_1, intermediate_result_step=20)

# -----------------------------------------------
# Visualization
print("============ Visualization ========")
visualization = Visualization(model)
visualization.plot_loss(loss_list)

# Prepare a video sample for intermediate results visualization
video = data_loader_1.dataset[4200][0]

# Set hooks to register intermediate results then perform a forward to pass the video through the model
visualization.set_hooks()
visualization.perform_forward(video=video)

# Visualize the video
visualization.plot_video(video)

# Visualize first CNN layer
visualization.plot_first_cnn_layer()

# Visualize the spatial bloc output
visualization.plot_spatial_bloc_output()

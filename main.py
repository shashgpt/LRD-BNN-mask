from scripts.all_imports import *
from scripts.preprocess_dataset import Preprocess_dataset
from scripts.train import Train


def mask_unused_gpus(leave_unmasked=1):
    """
    return the gpu no. with highest available memory
    """
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        available_gpus = [i for i, x in enumerate(memory_free_values)]
        if len(available_gpus) < leave_unmasked: raise ValueError('Found only %d usable GPUs in the system' % len(available_gpus))
        gpu_with_highest_free_memory = 0
        highest_free_memory = 0
        for index, memory in enumerate(memory_free_values):
            if memory > highest_free_memory:
                highest_free_memory = memory
                gpu_with_highest_free_memory = index
        return str(gpu_with_highest_free_memory)
    except Exception as e:
        print('"nvidia-smi" is probably not installed. GPUs are not masked', e)
# def set_cuda_device():
#     device_no_with_highest_free_mem = None
#     highest_free_memory = 0
#     for device_no in range(torch.cuda.device_count()):
#         nvmlInit()
#         handle = nvmlDeviceGetHandleByIndex(device_no)
#         info = nvmlDeviceGetMemoryInfo(handle)
#         free_mem = info.free/1000000000
#         if free_mem > highest_free_memory:
#             highest_free_memory = free_mem
#             device_no_with_highest_free_mem = device_no
#     return device_no_with_highest_free_mem


def main():
    """
    main driver function
    """

    # configuration parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_value",
                        type=int,
                        required=True,
                        help="The seed value for the code")
    parser.add_argument("--create_input_data",
                        type=bool,
                        required=True,
                        help="Whether to preprocess and create input data or just load the existing preprocessed dataset")
    parser.add_argument("--dataset_name",
                        type=str,
                        required=True,
                        help="Name of the dataset to be used")
    parser.add_argument("--dataset_dir",
                        type=str,
                        required=True,
                        help="Dataset path")
    parser.add_argument("--model_name",
                        type=str,
                        required=True,
                        help="Name of the model")  
    parser.add_argument("--train",
                        type=bool,
                        required=True,
                        help="To train a model")
    parser.add_argument("--word_embeddings",
                        type=str,
                        required=True,
                        help="Name of the pretrained word embeddings")
    parser.add_argument("--pretrained_word_embeddings_path",
                        type=str,
                        required=True,
                        help="path of the pretrained word embeddings")
    parser.add_argument("--fine_tune_word_embeddings",
                        type=bool,
                        required=True,
                        help="Fine-tune word embeddings during training")
    parser.add_argument("--create_embedding_mask",
                        type=bool,
                        required=True,
                        help="Create embedding mask during training")
    parser.add_argument("--embedding_dim",
                        type=int,
                        required=True,
                        help="Word embeddings dimensions")         
    parser.add_argument("--sequence_layer_units",
                        type=int,
                        required=True,
                        help="Hidden units in the sequence layer")
    parser.add_argument("--num_of_bayesian_samples",
                        type=int,
                        required=True,
                        help="No of samples in the Bayesian Inference")
    parser.add_argument("--train_epochs",
                        type=int,
                        required=True,
                        help="No of train epochs")
    parser.add_argument("--batch_size",
                        type=int,
                        required=True,
                        help="No of train epochs")
    args = parser.parse_args()

    # set the seed values
    os.environ['PYTHONHASHSEED']=str(args.seed_value)
    random.seed(args.seed_value)
    np.random.seed(args.seed_value)
    tf.random.set_seed(args.seed_value)

    # set the runtime directory
    os.chdir(os.getcwd())

    # set the gpu device with limited memory growth
    os.environ["CUDA_VISIBLE_DEVICES"] = mask_unused_gpus()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # os.environ["CUDA_VISIBLE_DEVICES"] = torch.device(set_cuda_device())
    # torch.cuda.set_device(set_cuda_device())

    # create input data
    preprocess_dataset_obj = Preprocess_dataset(args)
    if args.create_input_data == True:
        preprocessed_dataset = preprocess_dataset_obj.preprocess()
        pickle.dump(preprocessed_dataset, open(args.dataset_dir+"preprocessed_dataset.pickle", "wb"))
    else:
        preprocessed_dataset = pickle.load(open(args.dataset_dir+"preprocessed_dataset.pickle", "rb")) 

    # train model or evaluate model
    if args.train == True:
        print("\nTraining")
        Train(args).train_model(preprocessed_dataset)
    else:
        print("\nEvaluating")
        Train(args).evaluate_model(preprocessed_dataset)
    
    # save all the assets
    

if __name__=="__main__":
    main()
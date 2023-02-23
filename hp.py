
class hp:

    data_root = '../../data/cleaned'
    n_train_speakers = 20
    n_test_speakers = 2 # these are the zero-shot speakers
    sampling_rate = 16000

    # speaker embedding settings
    n_uttr_per_spk_embedding = 1
    speaker_embedding_dir = '../../data/sse_embeddings/'

    # train settings
    output_path = './outputs/test_run_022323/'
    device = 'cpu'
    len_crop = 128
    # changed batch size from 4 to 1
    bs = 4
    n_iters = 1000 #2300000 # much greater than the 100k in the paper
    lamb = 1
    mu = 1
    tb_log_interval = 10
    print_log_interval = 100

    lr = 1e-4 # according to github issues, no lr schedule is used

    seed = 100
    mel_shift = None
    mel_scale = None

    # mel spec resolution / time steps applied to other conditioning variables
    fft_length = 1024
    hop_length = 256

    # dimensions of one-hot conditioning vectors
    f0_dim = 256
    amp_dim = 256

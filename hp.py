from torch._C import _dispatch_has_computed_kernel_for_dispatch_key


class hp:

    data_root = '/scratch/cnoufi/TAVA/data/audio-multichannel'
    n_train_speakers = 30
    n_test_speakers = 3 # these are the zero-shot speakers
    sampling_rate = 16000
    original_sr = 48000

    # speaker embedding settings
    n_uttr_per_spk_embedding = 1
    speaker_embedding_dir = '/scratch/cnoufi/TAVA/data/SSE/'

    # train settings
    output_path = '/scratch/cnoufi/TAVA/codebase/paralang-vc/outputs/test_run_cm-matlab_030723_100kIters/'
    #'/scratch/cnoufi/TAVA/codebase/paralang-vc/outputs/test_run_cm-matlab_030223_30kIters/'
    device = 'cuda:0'
    len_crop = 128
    # changed batch size from 4 to 1
    bs = 4
    n_iters = 100000 #2300000 # much greater than the 100k in the paper
    lamb = 1
    mu = 1
    gamma = 1
    tb_log_interval = 10
    print_log_interval = 100
    media_log_interval = 1000

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
    sse_dim = 256 #fixed

    # model dims
    dim_neck = 32
    dim_pre = 512
    freq = 32

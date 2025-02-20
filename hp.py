from torch._C import _dispatch_has_computed_kernel_for_dispatch_key


class hp:

    data_root = '/usr/ccrma/media/projects/jordan/Datasets/TAVA/audio-multichannel' #or /scratch/ folder of current machine if data lives there as well.
    n_train_speakers = 30
    n_test_speakers = 3 # these are the zero-shot speakers
    sampling_rate = 16000
    original_sr = 48000

    # speaker embedding settings
    n_uttr_per_spk_embedding = 1
    speaker_embedding_dir = '/scratch/cnoufi/TAVA/data/SSE/'

    # train settings
    run_dir = 'e3_v1_040423'
        #Should have format: 
        #   e{experiment code from spreadsheet}_v{version if applicable}_MMDDYYYY  
        #   e.g.  e1_v2_030723
    output_path = '/usr/ccrma/media/projects/jordan/Datasets/TAVA/outputs'
    device = 'cuda:0'
    len_crop = 128
    bs = 2
    n_iters = 30000 #2300000 # much greater than the 100k in the paper
    
    alpha = 1 #EGG melspec loss weight
    beta = 1 #EGG SNR loss weight
    lamb = 1 #postnet loss weight
    mu = 1 #tEGG loss weight
    gamma = 1 #EGG bottleneck code loss weight


    tb_log_interval = 10
    print_log_interval = 100
    media_log_interval = 250

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

    #transform flags
    transform_dict = {'speed':True,
                      'gain':True,
                      'scramble':True,
                      'noise':True,
                      'reverse':False
                      }

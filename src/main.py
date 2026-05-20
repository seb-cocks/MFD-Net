# Main file for the MFD-Net repo

## MemMap Image Encoders (Not implemented) (This would require importing over the code from the AIMC-Spec repo)
# from src.STFT_CNN.STFT_CNN_MemMap import main as stft_main
# from src.LPI_Net.LPI_Net_MemMap import main as lpi_main
# from src.LDC_Unet.LDC_Unet_MemMap import main as ldc_main
# from src.CDAE_DCNN.CDAE_DCNN_MemMap import main as cdae_main
# from src.ViT.Vit_MemMap import main as vit_main

from src.ResNet.ResNet_MemMap import main as res_main

from src.mfd_stream_existing_iq.driver import main as stream_memmap_main

from src.mfd_stream_train.driver import main as mfd_stream_train_main

# python -m src.main

if __name__ == "__main__":

    ### To create the streaming MemMap data
    stream_memmap_main()

    ### To train up the ResNet18
    # res_main()

    ### To train the MFD-Net streaming
    # mfd_stream_train_main()

    ### AIMC-Spec Runs (Not Implemented)
    # stft_main()
    # ldc_main()
    # lpi_main()
    # cdae_main()
    # vit_main()

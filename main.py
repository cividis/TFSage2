import os
from tfsage.download import download_chip_atlas, download_encode

os.makedirs("downloads", exist_ok=True)
download_chip_atlas("SRX502813", "downloads/chip.bed")
download_encode("ENCFF729IPL", "downloads/encode.bed")

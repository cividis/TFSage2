import subprocess
import tempfile


def curl_and_sort(url: str, output_file: str) -> None:
    with tempfile.NamedTemporaryFile() as temp_file:
        curl_command = ["curl", "-L", url, "-o", temp_file.name]
        sort_command = ["sortBed", "-i", temp_file.name]

        subprocess.run(curl_command, check=True)
        subprocess.run(sort_command, check=True, stdout=open(output_file, "w"))

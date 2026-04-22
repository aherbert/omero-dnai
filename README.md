# omero-dnai

OMERO DNAi uses [DNAi](https://github.com/ClementPla/DNAi) to analyse images
in [OMERO](https://www.openmicroscopy.org/omero/).

## Installation

```bash
# Install uv
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows (not tested!)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone the repository
git clone https://github.com/aherbert/omero-dnai.git
# Change into the project directory
cd omero-dnai
# Create and activate virtual environment
uv sync
source .venv/bin/activate
```

## Analysis

The script will connect to an OMERO server using the provided credentials.
Analysis is performed using multi-channel 2D images with a single timepoint.
The first and second channel for analysis can be specified. `DNAi` requires
the pixel size in micrometres. This is obtained from OMERO image metadata.

Analysis of a single image, or all images in a dataset, uses the image
or dataset ID:

```
# Show options
uv run main.py -h

# Analyse
uv run main.py ID [ID ...]
```

Connection to OMERO uses the [OMERO.py](https://github.com/ome/omero-py)
Python bindings. The script will ask for your OMERO server
URL and username and password. Repeat invocations will reuse an active
session or reconnect if it has timed out.

Dataset ID takes precedence. If the ID for an image matches a dataset then
the dataset will be used. If no dataset if found for the ID then it is
assumed the ID is for an image.

Multiple IDs can be provided to analyse selected images.
The alternative is to put all images into a new OMERO dataset for
analysis. Note using the image copy functionality in OMERO does not duplicate
image data for a new dataset.

Results are saved to `results_[ID].csv`. The result file will not be overwritten
if it exists. The name can be changed or the file overwritten using options.

The defaults will filter fibres to those that can be classified, and discard
those above the error probability threshold. The threshold can be changed using
options.

## DNAi

The `DNAi` library will connect to [Hugging Face](https://huggingface.co/) to
download models. This may log warnings if not using an authenticated login
configured using environment variables.

The `DNAi` library uses some functionality of the [Streamlit](https://streamlit.io/)
which may log warnings. These can typically be ignored.

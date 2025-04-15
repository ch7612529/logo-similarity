# Logo Similarity Clustering

A system to match and group websites by the similarity of their logos without using traditional ML clustering algorithms.

## Overview

This project provides an end-to-end solution for:
1. Extracting logos from a list of websites
2. Generating visual fingerprints of these logos using perceptual hashing
3. Grouping websites based on logo similarity
4. Outputting clusters of websites with similar logos

The approach uses image processing techniques rather than machine learning algorithms like DBSCAN or k-means clustering, making it more interpretable and requiring no training data.

## Features

- **High extraction rate**: Successfully extracts logos from >97% of websites
- **Robust logo detection**: Uses multiple strategies to identify the most likely logo on each website
- **Perceptual hashing**: Compares logos based on visual similarity rather than exact pixel matching
- **Adaptive threshold selection**: Automatically determines the optimal similarity threshold for grouping
- **Parallel processing**: Uses concurrent execution for efficient processing of large domain lists
- **Detailed logging**: Provides comprehensive logs of the extraction and matching process

## Requirements

- Python 3.7+
- Pandas & PyArrow (for Parquet file handling)
- Requests & BeautifulSoup4 (for web scraping)
- Pillow & ImageHash (for image processing)
- NumPy & tqdm (for computation and progress tracking)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/logo-similarity-clustering.git
cd logo-similarity-clustering
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Input Data

The script expects a Parquet file named `logos_list.parquet` containing website domains to analyze. The file should have a column named "domain" (or similar) with domain names in the format "example.com".

If your data is in a different format, you can convert it to Parquet using pandas:
```python
import pandas as pd

# Read your data
df = pd.read_csv("your_domains.csv")  # or Excel, JSON, etc.

# Save as Parquet
df.to_parquet("logos_list.parquet")
```

### Running the Script

1. Place your `logos_list.parquet` file in the project directory
2. Run the script:
```bash
python logo_similarity_clustering.py
```

### Output

The script will generate:
1. `extracted_logos/` directory containing all downloaded logo images
2. `logo_groups.csv` with website groupings based on logo similarity
3. `logo_extraction.log` with detailed logs of the extraction process

## How It Works

### Step 1: Logo Extraction

The system employs multiple strategies to locate and extract logos from websites:
- Checking meta tags (OpenGraph, Twitter)
- Searching for images with "logo" in their attributes (src, alt, class, id)
- Analyzing image positioning (logos are typically in headers)
- Using favicons as a fallback

Each potential logo is scored and validated to ensure it meets logo criteria (appropriate size, color complexity, etc.).

### Step 2: Perceptual Hashing

Instead of comparing images pixel by pixel, the system uses perceptual hashing to create "fingerprints" that:
- Capture the visual essence of logos
- Are resistant to minor variations in size, color, and format
- Enable meaningful similarity comparisons

The default approach uses pHash (perceptual hash), but the system also supports other algorithms like aHash (average hash), dHash (difference hash), and wHash (wavelet hash).

### Step 3: Similarity Matching

The system:
1. Computes the similarity between all pairs of logo hashes
2. Tests multiple similarity thresholds to find the optimal balance
3. Groups logos using a graph-based approach where each group contains logos that are similar above the threshold

## Customization

You can modify key parameters in the script:
- `hash_method`: The perceptual hashing algorithm to use ('phash', 'dhash', 'ahash', 'whash')
- `hash_size`: Size of the hash (larger = more detail but slower)
- Similarity thresholds: The list of thresholds to test
- Extraction parameters: Various scoring weights and criteria for logo detection

## Scaling to Large Datasets

For very large datasets (thousands or millions of domains):
1. Increase the number of worker threads in the ThreadPoolExecutor
2. Consider implementing checkpointing to save progress
3. For extreme scale, adapt the code to use distributed processing frameworks

## Limitations

- The system requires internet access to scrape websites
- Some websites block automated access or have complex JavaScript-based rendering
- Very similar logos with different text elements may be grouped separately
- Very different logos from the same brand might not be grouped together

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

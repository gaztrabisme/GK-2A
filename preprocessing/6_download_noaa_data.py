"""
Download NOAA GOES-18 Satellite Imagery

Downloads thermal sandwich imagery from NOAA CDN for hurricane detection.
URL pattern: https://cdn.star.nesdis.noaa.gov/GOES18/ABI/FD/Sandwich/YYYYDDDHHM0_GOES18-ABI-FD-Sandwich-678x678.jpg

Where:
- YYYY: Year (e.g., 2024)
- DDD: Day of year (001-366)
- HH: Hour (00-23)
- M: Minute divided by 10 (0, 1, 2, 3, 4, 5 for 00, 10, 20, 30, 40, 50)
"""

import requests
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Tuple, Optional
import argparse


class NOAAImageDownloader:
    """Download GOES-18 ABI Full Disk Sandwich imagery from NOAA CDN"""

    BASE_URL = "https://cdn.star.nesdis.noaa.gov/GOES18/ABI/FD/Sandwich"

    def __init__(self, output_dir: str = "data/raw/noaa_downloads", verbose: bool = True):
        """
        Initialize downloader.

        Args:
            output_dir: Directory to save downloaded images
            verbose: Print progress messages
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        self.stats = {
            'total': 0,
            'downloaded': 0,
            'already_exists': 0,
            'failed': 0,
            'not_found': 0
        }

    def format_timestamp(self, dt: datetime) -> str:
        """
        Format datetime to NOAA filename format: YYYYDDDHHM0

        Args:
            dt: Datetime object

        Returns:
            Formatted string (e.g., "2024274123" for day 274, 12:30)
        """
        year = dt.strftime('%Y')
        day_of_year = dt.strftime('%j')  # Day of year (001-366)
        hour = dt.strftime('%H')
        minute_tens = str(dt.minute // 10)  # Convert minute to tens digit

        return f"{year}{day_of_year}{hour}{minute_tens}0"

    def build_url(self, dt: datetime) -> str:
        """
        Build full download URL for a given datetime.

        Args:
            dt: Datetime object

        Returns:
            Full URL to image
        """
        timestamp = self.format_timestamp(dt)
        filename = f"{timestamp}_GOES18-ABI-FD-Sandwich-678x678.jpg"
        return f"{self.BASE_URL}/{filename}"

    def download_image(self, dt: datetime, retry_count: int = 3, timeout: int = 30) -> Optional[Path]:
        """
        Download a single image with retry logic.

        Args:
            dt: Datetime to download
            retry_count: Number of retries on failure
            timeout: Request timeout in seconds

        Returns:
            Path to downloaded file, or None if failed
        """
        url = self.build_url(dt)
        timestamp = self.format_timestamp(dt)
        filename = f"{timestamp}_GOES18-ABI-FD-Sandwich-678x678.jpg"
        filepath = self.output_dir / filename

        # Check if already exists
        if filepath.exists():
            if self.verbose:
                print(f"✓ Already exists: {filename}")
            self.stats['already_exists'] += 1
            return filepath

        # Try downloading with retries
        for attempt in range(retry_count):
            try:
                response = requests.get(url, timeout=timeout, stream=True)

                if response.status_code == 200:
                    # Download successful
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    file_size_kb = filepath.stat().st_size / 1024
                    if self.verbose:
                        print(f"✓ Downloaded: {filename} ({file_size_kb:.1f} KB)")

                    self.stats['downloaded'] += 1
                    return filepath

                elif response.status_code == 404:
                    # Image not available (common for missing data)
                    if self.verbose:
                        print(f"✗ Not found: {filename}")
                    self.stats['not_found'] += 1
                    return None

                else:
                    # Other HTTP error
                    if attempt < retry_count - 1:
                        if self.verbose:
                            print(f"⚠ HTTP {response.status_code} for {filename}, retrying...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        if self.verbose:
                            print(f"✗ Failed: {filename} (HTTP {response.status_code})")
                        self.stats['failed'] += 1
                        return None

            except requests.exceptions.Timeout:
                if attempt < retry_count - 1:
                    if self.verbose:
                        print(f"⚠ Timeout for {filename}, retrying...")
                    time.sleep(2 ** attempt)
                else:
                    if self.verbose:
                        print(f"✗ Timeout: {filename}")
                    self.stats['failed'] += 1
                    return None

            except Exception as e:
                if attempt < retry_count - 1:
                    if self.verbose:
                        print(f"⚠ Error for {filename}: {str(e)}, retrying...")
                    time.sleep(2 ** attempt)
                else:
                    if self.verbose:
                        print(f"✗ Error: {filename} - {str(e)}")
                    self.stats['failed'] += 1
                    return None

        return None

    def generate_timestamps(self, start_date: datetime, end_date: datetime) -> list:
        """
        Generate all 10-minute interval timestamps between start and end dates.

        Args:
            start_date: Start datetime
            end_date: End datetime

        Returns:
            List of datetime objects at 10-minute intervals
        """
        # Round start to nearest 10 minutes
        start_minute = (start_date.minute // 10) * 10
        current = start_date.replace(minute=start_minute, second=0, microsecond=0)

        timestamps = []
        while current <= end_date:
            timestamps.append(current)
            current += timedelta(minutes=10)

        return timestamps

    def download_date_range(self, start_date: datetime, end_date: datetime,
                           delay: float = 0.1) -> dict:
        """
        Download all images in a date range.

        Args:
            start_date: Start datetime
            end_date: End datetime
            delay: Delay between requests in seconds (to be nice to server)

        Returns:
            Dictionary with download statistics
        """
        # Reset stats
        self.stats = {
            'total': 0,
            'downloaded': 0,
            'already_exists': 0,
            'failed': 0,
            'not_found': 0
        }

        timestamps = self.generate_timestamps(start_date, end_date)
        self.stats['total'] = len(timestamps)

        if self.verbose:
            print("="*80)
            print(f"NOAA GOES-18 IMAGE DOWNLOAD")
            print("="*80)
            print(f"Date range: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"Total timestamps: {len(timestamps)}")
            print(f"Output directory: {self.output_dir}")
            print("="*80)
            print()

        start_time = time.time()

        for i, dt in enumerate(timestamps):
            self.download_image(dt)

            # Progress update every 50 images
            if self.verbose and (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(timestamps) - i - 1) / rate if rate > 0 else 0
                print(f"\nProgress: {i+1}/{len(timestamps)} ({(i+1)/len(timestamps)*100:.1f}%)")
                print(f"  Downloaded: {self.stats['downloaded']}, Exists: {self.stats['already_exists']}, "
                      f"Missing: {self.stats['not_found']}, Failed: {self.stats['failed']}")
                print(f"  Rate: {rate:.1f} images/sec, ETA: {remaining/60:.1f} minutes\n")

            # Be nice to the server
            time.sleep(delay)

        # Final summary
        if self.verbose:
            elapsed = time.time() - start_time
            print("\n" + "="*80)
            print("DOWNLOAD COMPLETE")
            print("="*80)
            print(f"Total processed: {self.stats['total']}")
            print(f"  ✓ Downloaded: {self.stats['downloaded']}")
            print(f"  ✓ Already exists: {self.stats['already_exists']}")
            print(f"  ✗ Not found (404): {self.stats['not_found']}")
            print(f"  ✗ Failed: {self.stats['failed']}")
            print(f"Elapsed time: {elapsed/60:.1f} minutes")
            print(f"Success rate: {(self.stats['downloaded'] + self.stats['already_exists'])/self.stats['total']*100:.1f}%")
            print("="*80)

        return self.stats


# Predefined hurricane periods for 2024
HURRICANE_EVENTS_2024 = {
    'beryl': ('2024-06-25', '2024-07-12'),      # Hurricane Beryl
    'ernesto': ('2024-08-09', '2024-08-23'),    # Hurricane Ernesto
    'francine': ('2024-09-06', '2024-09-15'),   # Hurricane Francine
    'helene': ('2024-09-20', '2024-09-30'),     # Hurricane Helene
    'milton': ('2024-10-02', '2024-10-13'),     # Hurricane Milton
}


def download_hurricane_event(event_name: str, output_base_dir: str = "data/raw/noaa_2024"):
    """
    Download images for a specific hurricane event.

    Args:
        event_name: Name of hurricane event (e.g., 'beryl', 'milton')
        output_base_dir: Base directory for downloads
    """
    if event_name not in HURRICANE_EVENTS_2024:
        print(f"Error: Unknown event '{event_name}'")
        print(f"Available events: {', '.join(HURRICANE_EVENTS_2024.keys())}")
        return

    start_str, end_str = HURRICANE_EVENTS_2024[event_name]
    start_date = datetime.strptime(start_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_str, '%Y-%m-%d').replace(hour=23, minute=59)

    output_dir = Path(output_base_dir) / event_name
    downloader = NOAAImageDownloader(output_dir=output_dir, verbose=True)

    downloader.download_date_range(start_date, end_date)


def download_test_dataset(output_dir: str = "data/raw/noaa_test_2024"):
    """
    Download a 2-week test dataset from September 2024 (active hurricane season).

    Args:
        output_dir: Where to save test images
    """
    # September 6-20, 2024 (covers Hurricane Francine)
    start_date = datetime(2024, 9, 6, 0, 0)
    end_date = datetime(2024, 9, 20, 23, 59)

    downloader = NOAAImageDownloader(output_dir=output_dir, verbose=True)
    downloader.download_date_range(start_date, end_date)


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Download NOAA GOES-18 satellite imagery')

    parser.add_argument('--mode', choices=['test', 'event', 'custom'], default='test',
                       help='Download mode: test dataset, specific event, or custom date range')

    parser.add_argument('--event', type=str,
                       help='Hurricane event name (beryl, ernesto, francine, helene, milton)')

    parser.add_argument('--start', type=str,
                       help='Start date (YYYY-MM-DD HH:MM) for custom mode')

    parser.add_argument('--end', type=str,
                       help='End date (YYYY-MM-DD HH:MM) for custom mode')

    parser.add_argument('--output', type=str, default='data/raw/noaa_downloads',
                       help='Output directory')

    args = parser.parse_args()

    if args.mode == 'test':
        print("Downloading test dataset (Sep 6-20, 2024)...")
        download_test_dataset(output_dir=args.output)

    elif args.mode == 'event':
        if not args.event:
            print("Error: --event required for event mode")
            print(f"Available events: {', '.join(HURRICANE_EVENTS_2024.keys())}")
            return

        print(f"Downloading Hurricane {args.event.capitalize()}...")
        download_hurricane_event(args.event, output_base_dir=args.output)

    elif args.mode == 'custom':
        if not args.start or not args.end:
            print("Error: --start and --end required for custom mode")
            return

        try:
            # Try parsing with time
            try:
                start_date = datetime.strptime(args.start, '%Y-%m-%d %H:%M')
            except:
                start_date = datetime.strptime(args.start, '%Y-%m-%d')

            try:
                end_date = datetime.strptime(args.end, '%Y-%m-%d %H:%M')
            except:
                end_date = datetime.strptime(args.end, '%Y-%m-%d').replace(hour=23, minute=59)

        except ValueError as e:
            print(f"Error parsing dates: {e}")
            print("Format: YYYY-MM-DD or YYYY-MM-DD HH:MM")
            return

        print(f"Downloading custom range: {start_date} to {end_date}...")
        downloader = NOAAImageDownloader(output_dir=args.output, verbose=True)
        downloader.download_date_range(start_date, end_date)


if __name__ == "__main__":
    # Example usage without command line args
    import sys

    if len(sys.argv) == 1:
        # No arguments - run test download
        print("No arguments provided. Running test download...")
        print("For custom usage, run with --help")
        print()
        download_test_dataset()
    else:
        main()

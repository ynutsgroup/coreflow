#!/usr/bin/env python3
from __future__ import annotations
import lzma
import struct
import httpx
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
from typing import Optional, Tuple

# Konfiguration
SYMBOL = "EURUSD"
DATE_RANGE = (datetime(2025, 7, 23), datetime(2025, 7, 25))
BASE_DIR = Path("/opt/coreflow/data/dukascopy")
RAW_DIR = BASE_DIR / "bi5_raw"
CSV_DIR = BASE_DIR / "csv_ticks"

# Logger Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "tick_processor.log"),
        logging.StreamHandler()
    ]
)

class DukascopyTickProcessor:
    """Moderne Implementierung für Dukascopy BI5 Tick-Daten"""
    
    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "CoreFlow/1.0"},
            follow_redirects=True
        )
        self._setup_dirs()

    def _setup_dirs(self) -> None:
        """Erstellt benötigte Verzeichnisse"""
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        CSV_DIR.mkdir(parents=True, exist_ok=True)

    def _get_bi5_url(self, date: datetime, hour: int) -> str:
        """Generiert die korrekte Dukascopy-URL mit 0-basiertem Monat"""
        return (
            f"https://datafeed.dukascopy.com/datafeed/"
            f"{SYMBOL}/{date.year:04}/"
            f"{date.month - 1:02}/"  # Wichtig: Monat -1
            f"{date.day:02}/"
            f"{hour:02}h_ticks.bi5"
        )

    def _fetch_bi5(self, url: str) -> Optional[bytes]:
        """Lädt BI5-Daten herunter"""
        try:
            response = self.client.get(url)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as e:
            logging.warning(f"Download failed for {url}: {e}")
            return None

    def _parse_ticks(self, data: bytes) -> pd.DataFrame:
        """Dekodiert BI5-Daten zu Tick-Daten"""
        try:
            decompressed = lzma.decompress(data)
            ticks = [
                {
                    "timestamp": timestamp / 1000,  # Unix timestamp in seconds
                    "ask": ask,
                    "bid": bid,
                    "volume": volume
                }
                for timestamp, ask, bid, volume in struct.iter_unpack(">Q3d", decompressed)
            ]
            return pd.DataFrame(ticks)
        except (lzma.LZMAError, struct.error) as e:
            logging.error(f"Parse error: {e}")
            return pd.DataFrame()

    def _save_ticks(self, df: pd.DataFrame, date: datetime, hour: int) -> bool:
        """Speichert Ticks im Parquet-Format"""
        try:
            csv_path = CSV_DIR / f"{SYMBOL}_{date:%Y%m%d}_{hour:02}.parquet"
            df.to_parquet(csv_path, compression="zstd")
            return True
        except Exception as e:
            logging.error(f"Save failed: {e}")
            return False

    def process_hour(self, date: datetime, hour: int) -> Tuple[bool, str]:
        """Verarbeitet eine einzelne Stunde"""
        url = self._get_bi5_url(date, hour)
        
        if (bi5_data := self._fetch_bi5(url)) is None:
            return False, f"Download failed for {url}"
            
        df = self._parse_ticks(bi5_data)
        if df.empty:
            return False, f"No valid ticks in {url}"
            
        if not self._save_ticks(df, date, hour):
            return False, f"Save failed for {url}"
            
        return True, f"Processed {len(df)} ticks from {url}"

    def run(self):
        """Hauptverarbeitungsroutine"""
        date = DATE_RANGE[0]
        total_hours = (DATE_RANGE[1] - DATE_RANGE[0]).days * 24
        
        with tqdm(total=total_hours, desc="Processing hours") as pbar:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                
                while date <= DATE_RANGE[1]:
                    for hour in range(24):
                        futures.append(
                            executor.submit(self.process_hour, date, hour)
                        )
                    date += timedelta(days=1)
                
                for future in futures:
                    success, message = future.result()
                    if success:
                        pbar.update(1)
                    else:
                        logging.warning(message)

if __name__ == "__main__":
    processor = DukascopyTickProcessor()
    processor.run()
    logging.info("Processing completed")

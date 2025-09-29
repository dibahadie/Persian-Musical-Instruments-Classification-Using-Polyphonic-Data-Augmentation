import sys
import os
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QLabel, QCheckBox, QFileDialog, QMessageBox
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl

INSTRUMENTS = ["tonbak", "daat", "avaz", "piano", "violin",
               "tar", "sitar", "santur", "kamancheh", "ney"]

class AudioTagger(QWidget):
    def __init__(self, chunks_dir, csv_path):
        super().__init__()
        self.chunks_dir = Path(chunks_dir)
        self.csv_path = Path(csv_path)

        # collect all chunks
        import re

        def natural_key(path: Path):
            """Extract number after 'sample' for numeric sort, fallback to name."""
            m = re.search(r"sample(\d+)", path.stem)
            return int(m.group(1)) if m else 1e9  # put non-matching at the end

        self.chunks = sorted(
            [p for p in self.chunks_dir.rglob("sample*") if p.suffix.lower() in [".wav", ".mp3"]],
            key=natural_key
        )

        if not self.chunks:
            raise RuntimeError(f"No chunks found under {chunks_dir}")

        # load or init dataframe
        if self.csv_path.exists():
            self.df = pd.read_csv(self.csv_path)
        else:
            cols = ["sample_id"] + INSTRUMENTS
            self.df = pd.DataFrame(columns=cols)

        self.idx = 0
        self.init_ui()
        self.load_chunk(0)

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("File: ")
        layout.addWidget(self.label)

        # audio player
        self.player = QMediaPlayer()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_audio)
        layout.addWidget(self.play_btn)

        # instrument checkboxes
        self.checkboxes = {}
        check_layout = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()
        for i, inst in enumerate(INSTRUMENTS):
            cb = QCheckBox(inst)
            self.checkboxes[inst] = cb
            (left if i < len(INSTRUMENTS)//2 else right).addWidget(cb)
        check_layout.addLayout(left)
        check_layout.addLayout(right)
        layout.addLayout(check_layout)

        # navigation buttons
        nav = QHBoxLayout()
        prev_btn = QPushButton("Prev")
        prev_btn.clicked.connect(self.prev_chunk)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_current)
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(self.next_chunk)
        nav.addWidget(prev_btn)
        nav.addWidget(save_btn)
        nav.addWidget(next_btn)
        layout.addLayout(nav)

        self.setLayout(layout)
        self.setWindowTitle("Audio Chunk Tagger")

    def load_chunk(self, idx):
        if idx < 0 or idx >= len(self.chunks):
            return
        self.idx = idx
        f = self.chunks[idx]
        self.label.setText(f"File {idx+1}/{len(self.chunks)}: {f.name}")

        # set media
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(str(f))))

        # restore saved labels if exist
        sid = str(f.relative_to(self.chunks_dir))
        row = self.df[self.df["sample_id"] == sid]
        if not row.empty:
            row = row.iloc[0]
            for inst in INSTRUMENTS:
                self.checkboxes[inst].setChecked(bool(row.get(inst, 0)))
        else:
            for cb in self.checkboxes.values():
                cb.setChecked(False)

    def play_audio(self):
        self.player.stop()
        self.player.play()

    def save_current(self):
        f = self.chunks[self.idx]
        sid = str(f.relative_to(self.chunks_dir))
        data = {"sample_id": sid}
        for inst, cb in self.checkboxes.items():
            data[inst] = 1 if cb.isChecked() else 0
        # update or append
        if (self.df["sample_id"] == sid).any():
            self.df.loc[self.df["sample_id"] == sid, INSTRUMENTS] = [data[i] for i in INSTRUMENTS]
        else:
            self.df.loc[len(self.df)] = data
        self.df.to_csv(self.csv_path, index=False)
        QMessageBox.information(self, "Saved", f"Labels saved for {sid}")

    def next_chunk(self):
        if self.idx < len(self.chunks)-1:
            self.save_current()
            self.load_chunk(self.idx+1)

    def prev_chunk(self):
        if self.idx > 0:
            self.save_current()
            self.load_chunk(self.idx-1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    chunks_dir = QFileDialog.getExistingDirectory(None, "Select chunks directory", str(Path.cwd()))
    if not chunks_dir:
        sys.exit(0)
    csv_path, _ = QFileDialog.getSaveFileName(None, "Select CSV file", str(Path.cwd() / "tags.csv"), "CSV Files (*.csv)")
    if not csv_path:
        sys.exit(0)
    window = AudioTagger(chunks_dir, csv_path)
    window.show()
    sys.exit(app.exec_())

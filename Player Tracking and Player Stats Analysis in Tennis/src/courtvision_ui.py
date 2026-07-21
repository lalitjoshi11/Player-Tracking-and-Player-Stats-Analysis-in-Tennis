# ─────────────────────────────────────────────────────────────────────────────
#  UI — fetch a YouTube match clip and run analysis
#
# Requirements:
#   pip install yt-dlp
#   ffmpeg must be installed and on PATH
#       Download: https://ffmpeg.org/download.html
#       Windows:  winget install ffmpeg
#
# Run:
#   python courtvision_ui.py
# ─────────────────────────────────────────────────────────────────────────────

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import subprocess
import os
import sys
import re

# ── Output paths ──────────────────────────────────────────────────────────────
CLIP_OUTPUT_DIR = r"D:\NOTES\VI semester\Minor Project\MP"          # where clipped videos are saved
ANALYSIS_OUTPUT = r"D:\NOTES\VI semester\Minor Project\output.mp4"  # main.py output path
MAIN_PY_PATH    = os.path.join(os.path.dirname(__file__), r"D:\NOTES\VI semester\Minor Project\MP\main\new\main.py")

# ── Colour palette ────────────────────────────────────────────────────────────
BG          = "#000000"      # deep navy background
PANEL       = "#111827"      # card background
BORDER      = "#1e2d40"      # subtle border"
ACCENT      = "#00d4ff"      # cyan accent
ACCENT2     = "#00ff88"      # green accent
TEXT        = "#e8f0fe"      # primary text
TEXT_DIM    = "#6b7a99"      # secondary text
RED         = "#ff4757"      # error red
YELLOW      = "#ffd32a"      # warning yellow

FONT_TITLE  = ("Courier New", 22, "bold")
FONT_HEAD   = ("Courier New", 11, "bold")
FONT_BODY   = ("Courier New", 10)
FONT_SMALL  = ("Courier New", 9)
FONT_LOG    = ("Courier New", 9)

# ── Tool paths ────────────────────────────────────────────────────────────────
FFMPEG_PATH = r"D:\NOTES\VI semester\Minor Project\MP\main\ffmpeg\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

# ── Helpers ───────────────────────────────────────────────────────────────────

def time_to_seconds(h, m, s):
    return int(h) * 3600 + int(m) * 60 + int(s)


def seconds_to_hhmmss(total):
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def check_dependencies():
    """Check yt-dlp and ffmpeg are available."""
    issues = []
    try:
        import yt_dlp
    except ImportError:
        issues.append("yt-dlp not installed — run: pip install yt-dlp")
    try:
        subprocess.run(["ffmpeg", "-version"],
                       capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("ffmpeg not found — install from https://ffmpeg.org/download.html")
    return issues


# ── Main App ──────────────────────────────────────────────────────────────────

class PlayerStatsAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plyer Stats Analyzer")
        self.root.configure(bg=BG)
        self.root.geometry("780x820")
        self.root.resizable(False, False)

        self._clip_path  = None   # path to clipped video
        self._running    = False  # analysis running flag

        self._build_ui()
        self._check_deps_on_start()

    # ── Dependency check ──────────────────────────────────────────────────────

    def _check_deps_on_start(self):
        issues = check_dependencies()
        if issues:
            for issue in issues:
                self._log(f"[WARN] {issue}", color=YELLOW)

    # ── UI Builder ────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=30, pady=(28, 0))

        tk.Label(header, text="◈ PlayerStats", font=FONT_TITLE,
                 bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(header, text="Analyzer", font=FONT_TITLE,
                 bg=BG, fg=ACCENT2).pack(side="left")


        self._divider()

        # ── Step 1: YouTube URL ──────────────────────────────────────────────
        self._section("MATCH SOURCE")

        url_frame = tk.Frame(self.root, bg=PANEL, bd=0,
                             highlightthickness=1,
                             highlightbackground=BORDER)
        url_frame.pack(fill="x", padx=30, pady=(0, 6))

        tk.Label(url_frame, text="YouTube URL",
                 font=FONT_SMALL, bg=PANEL, fg=TEXT_DIM).pack(anchor="w", padx=12, pady=(8,2))

        self.url_var = tk.StringVar()
        url_entry = tk.Entry(url_frame, textvariable=self.url_var,
                             font=FONT_BODY, bg="#0d1526", fg=TEXT,
                             insertbackground=ACCENT,
                             relief="flat", bd=0)
        url_entry.pack(fill="x", padx=12, pady=(0,10), ipady=6)

        # Paste button
        paste_btn = self._flat_btn(url_frame, "  PASTE FROM CLIPBOARD  ",
                                   self._paste_url, BORDER, TEXT_DIM)
        paste_btn.pack(anchor="w", padx=12, pady=(0,10))

        self._divider()

        # ── Step 2: Time Range ───────────────────────────────────────────────
        self._section("TIME RANGE")

        time_frame = tk.Frame(self.root, bg=PANEL, bd=0,
                              highlightthickness=1,
                              highlightbackground=BORDER)
        time_frame.pack(fill="x", padx=30, pady=(0, 6))

        inner = tk.Frame(time_frame, bg=PANEL)
        inner.pack(fill="x", padx=12, pady=12)

        # Start time
        tk.Label(inner, text="START TIME", font=FONT_SMALL,
                 bg=PANEL, fg=TEXT_DIM).grid(row=0, column=0, sticky="w", pady=(0,4))
        tk.Label(inner, text="DURATION", font=FONT_SMALL,
                 bg=PANEL, fg=TEXT_DIM).grid(row=0, column=2, sticky="w", pady=(0,4), padx=(40,0))

        # Start time spinners
        start_f = tk.Frame(inner, bg=PANEL)
        start_f.grid(row=1, column=0, sticky="w")

        self.start_h = self._spinner(start_f, 0, 23, 0)
        tk.Label(start_f, text="h", font=FONT_SMALL, bg=PANEL,
                 fg=TEXT_DIM).pack(side="left", padx=(2,8))
        self.start_m = self._spinner(start_f, 0, 59, 0)
        tk.Label(start_f, text="m", font=FONT_SMALL, bg=PANEL,
                 fg=TEXT_DIM).pack(side="left", padx=(2,8))
        self.start_s = self._spinner(start_f, 0, 59, 0)
        tk.Label(start_f, text="s", font=FONT_SMALL, bg=PANEL,
                 fg=TEXT_DIM).pack(side="left", padx=(2,0))

        # Duration quick-select buttons
        dur_f = tk.Frame(inner, bg=PANEL)
        dur_f.grid(row=1, column=2, sticky="w", padx=(40,0))

        self.duration_var = tk.IntVar(value=30)
        durations = [10, 20, 30, 45, 60, 90, 120]
        for d in durations:
            rb = tk.Radiobutton(dur_f, text=f"{d}s",
                                variable=self.duration_var, value=d,
                                font=FONT_SMALL, bg=PANEL, fg=TEXT,
                                selectcolor=BG,
                                activebackground=PANEL,
                                activeforeground=ACCENT,
                                indicatoron=0,
                                relief="flat", bd=0,
                                highlightthickness=1,
                                highlightbackground=BORDER,
                                padx=8, pady=4,
                                cursor="hand2",
                                command=self._update_preview)
            rb.pack(side="left", padx=3)

        # Custom duration
        tk.Label(inner, text="or custom (s):", font=FONT_SMALL,
                 bg=PANEL, fg=TEXT_DIM).grid(row=2, column=2, sticky="w",
                                              padx=(40,0), pady=(8,0))
        self.custom_dur_var = tk.StringVar()
        custom_entry = tk.Entry(inner, textvariable=self.custom_dur_var,
                                font=FONT_BODY, bg="#0d1526", fg=TEXT,
                                insertbackground=ACCENT,
                                relief="flat", bd=0, width=6)
        custom_entry.grid(row=3, column=2, sticky="w", padx=(40,0))
        self.custom_dur_var.trace_add("write", lambda *_: self._update_preview())

        # Preview label
        self.preview_var = tk.StringVar(value="Clip:  00:00:00  →  00:00:30  (30s)")
        tk.Label(inner, textvariable=self.preview_var,
                 font=FONT_SMALL, bg=PANEL, fg=ACCENT).grid(
                     row=4, column=0, columnspan=4, sticky="w", pady=(12,0))

        # Bind spinner changes to preview update
        for var in [self.start_h, self.start_m, self.start_s]:
            var.trace_add("write", lambda *_: self._update_preview())

        self._divider()

        # ── Step 3: Output ───────────────────────────────────────────────────
        self._section("OUTPUT PATH")

        out_frame = tk.Frame(self.root, bg=PANEL, bd=0,
                             highlightthickness=1,
                             highlightbackground=BORDER)
        out_frame.pack(fill="x", padx=30, pady=(0, 6))

        out_inner = tk.Frame(out_frame, bg=PANEL)
        out_inner.pack(fill="x", padx=12, pady=10)

        self.output_var = tk.StringVar(value=ANALYSIS_OUTPUT)
        out_entry = tk.Entry(out_inner, textvariable=self.output_var,
                             font=FONT_BODY, bg="#0d1526", fg=TEXT,
                             insertbackground=ACCENT,
                             relief="flat", bd=0)
        out_entry.pack(side="left", fill="x", expand=True, ipady=5)

        browse_btn = self._flat_btn(out_inner, " BROWSE ",
                                    self._browse_output, BORDER, TEXT_DIM)
        browse_btn.pack(side="left", padx=(8,0))

        self._divider()

        # ── Action buttons ───────────────────────────────────────────────────
        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(fill="x", padx=30, pady=(8, 0))

        self.fetch_btn = self._action_btn(
            btn_frame, "⬇  FETCH & CLIP", self._fetch_clip, ACCENT)
        self.fetch_btn.pack(side="left", padx=(0, 10))

        self.run_btn = self._action_btn(
            btn_frame, "▶  RUN ANALYSIS", self._run_analysis, ACCENT2)
        self.run_btn.pack(side="left")
        self.run_btn.config(state="disabled")

        self.play_btn = self._action_btn(
            btn_frame, "▶  PLAY OUTPUT", self._play_output, YELLOW)
        self.play_btn.pack(side="left", padx=(10, 0))
        self.play_btn.config(state="disabled")   # disabled until analysis done

        self.cancel_btn = self._action_btn(
            btn_frame, "✕  CANCEL", self._cancel, RED)
        self.cancel_btn.pack(side="right")
        self.cancel_btn.config(state="disabled")

        # ── Status bar ───────────────────────────────────────────────────────
        status_frame = tk.Frame(self.root, bg=PANEL, bd=0,
                                highlightthickness=1,
                                highlightbackground=BORDER)
        status_frame.pack(fill="x", padx=30, pady=(14, 0))

        self.status_var = tk.StringVar(value="ready")
        tk.Label(status_frame, textvariable=self.status_var,
                 font=FONT_SMALL, bg=PANEL, fg=ACCENT2,
                 anchor="w").pack(side="left", padx=12, pady=6)

        self.progress = ttk.Progressbar(status_frame, mode="indeterminate",
                                        length=120)
        self.progress.pack(side="right", padx=12, pady=8)

        # ── Log window ───────────────────────────────────────────────────────
        self._section("LOG")

        log_frame = tk.Frame(self.root, bg=PANEL, bd=0,
                             highlightthickness=1,
                             highlightbackground=BORDER)
        log_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))

        self.log_text = tk.Text(log_frame, font=FONT_LOG,
                                bg="#080c18", fg=TEXT_DIM,
                                insertbackground=ACCENT,
                                relief="flat", bd=0,
                                state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)

        # Log colour tags
        self.log_text.tag_config("cyan",   foreground=ACCENT)
        self.log_text.tag_config("green",  foreground=ACCENT2)
        self.log_text.tag_config("yellow", foreground=YELLOW)
        self.log_text.tag_config("red",    foreground=RED)
        self.log_text.tag_config("dim",    foreground=TEXT_DIM)

        self._log("PlayerStatsAnalyzer ready.", color=ACCENT2)

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _divider(self):
        tk.Frame(self.root, bg=BORDER, height=1).pack(
            fill="x", padx=30, pady=10)

    def _section(self, title):
        tk.Label(self.root, text=title, font=FONT_HEAD,
                 bg=BG, fg=TEXT_DIM).pack(anchor="w", padx=30, pady=(4, 6))

    def _spinner(self, parent, from_, to, initial):
        var = tk.StringVar(value=str(initial).zfill(2))
        sb = tk.Spinbox(parent, from_=from_, to=to, width=3,
                        textvariable=var, format="%02.0f",
                        font=FONT_BODY, bg="#0d1526", fg=TEXT,
                        buttonbackground=BORDER,
                        insertbackground=ACCENT,
                        relief="flat", bd=0)
        sb.pack(side="left")
        return var

    def _flat_btn(self, parent, text, cmd, bg, fg):
        return tk.Button(parent, text=text, command=cmd,
                         font=FONT_SMALL, bg=bg, fg=fg,
                         relief="flat", bd=0, padx=10, pady=4,
                         cursor="hand2",
                         activebackground=ACCENT, activeforeground=BG)

    def _action_btn(self, parent, text, cmd, accent):
        return tk.Button(parent, text=text, command=cmd,
                         font=FONT_HEAD, bg=accent, fg=BG,
                         relief="flat", bd=0, padx=18, pady=10,
                         cursor="hand2",
                         activebackground=TEXT, activeforeground=BG)

    # ── UI logic ──────────────────────────────────────────────────────────────

    def _paste_url(self):
        try:
            self.url_var.set(self.root.clipboard_get())
        except tk.TclError:
            pass

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi")])
        if path:
            self.output_var.set(path)

    def _get_duration(self):
        """Returns duration in seconds — custom field takes priority."""
        custom = self.custom_dur_var.get().strip()
        if custom.isdigit() and int(custom) > 0:
            return int(custom)
        return self.duration_var.get()

    def _update_preview(self):
        try:
            start = time_to_seconds(
                self.start_h.get() or 0,
                self.start_m.get() or 0,
                self.start_s.get() or 0
            )
            dur  = self._get_duration()
            end  = start + dur
            self.preview_var.set(
                f"Clip:  {seconds_to_hhmmss(start)}  →  "
                f"{seconds_to_hhmmss(end)}  ({dur}s)"
            )
        except Exception:
            pass

    def _log(self, message, color=None):
        """Append a line to the log window."""
        tag_map = {
            ACCENT:   "cyan",
            ACCENT2:  "green",
            YELLOW:   "yellow",
            RED:      "red",
            TEXT_DIM: "dim",
        }
        tag = tag_map.get(color, None)

        self.log_text.config(state="normal")
        if tag:
            self.log_text.insert("end", message + "\n", tag)
        else:
            self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _set_status(self, text, color=ACCENT2):
        self.status_var.set(text)

    def _set_busy(self, busy):
        self._running = busy
        if busy:
            self.progress.start(12)
            self.fetch_btn.config(state="disabled")
            self.run_btn.config(state="disabled")
            self.cancel_btn.config(state="normal")
        else:
            self.progress.stop()
            self.fetch_btn.config(state="normal")
            self.cancel_btn.config(state="disabled")

    # ── Fetch & Clip ──────────────────────────────────────────────────────────

    def _fetch_clip(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning("Missing URL", "Please enter a YouTube URL.")
            return
        if not re.match(r"https?://(www\.)?(youtube\.com|youtu\.be)/", url):
            messagebox.showwarning("Invalid URL", "Please enter a valid YouTube URL.")
            return

        self._set_busy(True)
        threading.Thread(target=self._fetch_thread, args=(url,),
                         daemon=True).start()
    
    def _fetch_thread(self, url):
        try:
            import yt_dlp

            # Verify ffmpeg exists before doing anything
            if not os.path.exists(FFMPEG_PATH):
                self._log(f"[ERROR] ffmpeg not found at: {FFMPEG_PATH}", color=RED)
                self._log("[ERROR] Update FFMPEG_PATH in courtvision_ui.py", color=RED)
                self._set_busy(False)
                return

            os.makedirs(CLIP_OUTPUT_DIR, exist_ok=True)

            start    = time_to_seconds(
                self.start_h.get() or 0,
                self.start_m.get() or 0,
                self.start_s.get() or 0
            )
            duration = self._get_duration()

            self._log(f"[INFO] URL: {url}", TEXT_DIM)
            self._log(f"[INFO] Start: {seconds_to_hhmmss(start)} | Duration: {duration}s", TEXT_DIM)
            self._set_status("downloading 1080p video...")

            # ── Step 1: Download full video at 1080p via yt-dlp ──────────
            self._log("[INFO] Downloading 1080p stream (this may take a moment)...", color=ACCENT)

            temp_path = os.path.join(CLIP_OUTPUT_DIR, "temp_download.mp4")
            ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
            self._log(f"[INFO] ffmpeg dir: {ffmpeg_dir}", TEXT_DIM)
            self._log(f"[INFO] ffmpeg exists: {os.path.exists(FFMPEG_PATH)}", TEXT_DIM)
            
            ydl_opts = {
                'format'             : 'bestvideo[height>=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height>=720][ext=mp4]+bestaudio[ext=m4a]/best',
                'outtmpl'            : temp_path,
                'quiet'              : False,
                'no_warnings'        : False,
                'merge_output_format': 'mp4',
                'ffmpeg_location'    : r"D:\NOTES\VI semester\Minor Project\MP\main\ffmpeg\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe",  # tell yt-dlp where ffmpeg is
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info  = ydl.extract_info(url, download=True)   # actually download
                title = info.get('title', 'clip')

            self._log(f"[INFO] Title: {title}", TEXT_DIM)
            self._log(f"[INFO] Download complete: {temp_path}", TEXT_DIM)

            # ── Step 2: Clip the downloaded file with ffmpeg ──────────────
            self._log("[INFO] Clipping with ffmpeg...", color=ACCENT)
            self._set_status("clipping video...")

            safe_title  = re.sub(r'[^\w-]', '_', title)[:40].strip('_')
            clip_name   = f"{safe_title}_{seconds_to_hhmmss(start).replace(':','-')}_{duration}s.mp4"
            clip_path   = os.path.join(CLIP_OUTPUT_DIR, clip_name)

            ffmpeg_cmd = [
                FFMPEG_PATH, "-y",
                "-ss", str(start),
                "-i", temp_path,
                "-t", str(duration),
                "-c:v", "libx264",   # ← re-encode to fix keyframe issue
                "-c:a", "aac",
                "-crf", "18",        # ← high quality (0=lossless, 51=worst, 18=visually lossless)
                "-preset", "fast",   # ← fast encoding
                clip_path
            ]
            self._log(f"[CMD] ffmpeg clip → {clip_path}", TEXT_DIM)

            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            self._proc = proc

            # Stream ffmpeg output to log
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self._log(line, TEXT_DIM)

            proc.wait()

            if proc.returncode != 0:
                self._log("[ERROR] ffmpeg failed. See log above.", color=RED)
                self._set_status("clip failed", RED)
                self._set_busy(False)
                return

            # ── Step 3: Delete temp file ──────────────────────────────────
            if os.path.exists(temp_path):
                os.remove(temp_path)
                self._log("[INFO] Temp file cleaned up.", TEXT_DIM)

            # ── Step 4: Verify clip resolution via ffprobe ────────────────
            ffprobe_path = FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe")
            if os.path.exists(ffprobe_path):
                probe = subprocess.run(
                    [ffprobe_path, "-v", "error",
                     "-select_streams", "v:0",
                     "-show_entries", "stream=width,height",
                     "-of", "csv=p=0", clip_path],
                    capture_output=True, text=True
                )
                res = probe.stdout.strip()
                if res:
                    self._log(f"[INFO] Clip resolution: {res}", color=ACCENT2)

            self._clip_path = clip_path
            self._log(f"[DONE] Clip saved: {clip_path}", color=ACCENT2)
            self._set_status(f"clip ready — {duration}s", ACCENT2)

            # Enable run button
            self.root.after(0, lambda: self.run_btn.config(state="normal"))

        except ImportError:
            self._log("[ERROR] yt-dlp not installed. Run: pip install yt-dlp", color=RED)
        except Exception as e:
            self._log(f"[ERROR] {e}", color=RED)
            self._set_status("error", RED)
        finally:
            self.root.after(0, lambda: self._set_busy(False))
    # ── Run Analysis ──────────────────────────────────────────────────────────

    def _run_analysis(self):
        if not self._clip_path or not os.path.exists(self._clip_path):
            messagebox.showwarning("No Clip", "Please fetch a clip first.")
            return

        output_path = self.output_var.get().strip()
        if not output_path:
            messagebox.showwarning("No Output", "Please set an output path.")
            return

        self._set_busy(True)
        threading.Thread(target=self._analysis_thread,
                         args=(self._clip_path, output_path),
                         daemon=True).start()

    def _analysis_thread(self, clip_path, output_path):
        try:
            self._log(f"[INFO] Input:  {clip_path}", TEXT_DIM)
            self._log(f"[INFO] Output: {output_path}", TEXT_DIM)
            self._log("[INFO] Starting Player Stats Analysis...", color=ACCENT)
            self._set_status("running analysis...")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Patch VIDEO_PATH and OUTPUT_PATH in main.py via env variables
            env = os.environ.copy()
            env["CV_VIDEO_PATH"]  = clip_path
            env["CV_OUTPUT_PATH"] = output_path

            proc = subprocess.Popen(
                [sys.executable, MAIN_PY_PATH],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env
            )
            self._proc = proc

            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                # Colour-code log lines
                if "[ERROR]" in line or "Error" in line:
                    self._log(line, color=RED)
                elif "[DONE]" in line or "Done" in line:
                    self._log(line, color=ACCENT2)
                elif "[TRACK]" in line:
                    self._log(line, color=ACCENT)
                else:
                    self._log(line, TEXT_DIM)

            proc.wait()

            if proc.returncode == 0:
                self._log(f"[DONE] Analysis complete → {output_path}", color=ACCENT2)
                self._set_status("analysis complete ✓", ACCENT2)
                #enable play button
                self.root.after(0, lambda: self.play_btn.config(state="normal"))
            else:
                self._log("[ERROR] Analysis failed. Check log above.", color=RED)
                self._set_status("analysis failed", RED)

        except Exception as e:
            self._log(f"[ERROR] {e}", color=RED)
        finally:
            self.root.after(0, lambda: self._set_busy(False))

    # ── Cancel ────────────────────────────────────────────────────────────────

    def _cancel(self):
        if hasattr(self, '_proc') and self._proc:
            self._proc.terminate()
            self._log("[INFO] Cancelled.", color=YELLOW)
            self._set_status("cancelled", YELLOW)
        self._set_busy(False)

    # ── Play Output ───────────────────────────────────────────────────────────   ← add here

    def _play_output(self):
        """Open the analysed output video in the default media player."""
        output_path = self.output_var.get().strip()
        if not output_path or not os.path.exists(output_path):
            messagebox.showwarning("No Output", "No analysed video found. Run analysis first.")
            return
        self._log(f"[INFO] Opening: {output_path}", color=ACCENT)
        os.startfile(output_path)   # opens with default media player on Windows
        
# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()

    # Style ttk progressbar
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TProgressbar",
                    troughcolor=BORDER,
                    background=ACCENT,
                    thickness=6)

    app = PlayerStatsAnalyzerApp(root)
    root.mainloop()

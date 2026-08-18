# Simple MP3/WAV Player Specification

## Project Overview
- **Name**: SimpleMusicPlayer
- **Type**: Desktop Music Player Application
- **Core Functionality**: A lightweight GUI-based music player that can browse file systems, manage playlists, and play MP3/WAV audio files
- **Target Users**: Users who want a basic, clean music player without bloat

## Technology Stack
- **Language**: Python 3
- **GUI Framework**: PyQt6
- **Audio Playback**: pygame (for reliable audio playback)
- **File**: single Python file, no external dependencies except PyQt6 and pygame

## UI/UX Specification

### Layout Structure
```
+----------------------------------------------------------+
|  [Menu Bar: File | Playlist | Help]                      |
+----------------------------------------------------------+
|  +------------------------+  +-------------------------+ |
|  |   FILE BROWSER        |  |      PLAYLIST           | |
|  |   (Left Panel)        |  |      (Right Panel)      | |
|  |                       |  |                         | |
|  |   [Tree View of       |  |   [Sort: Artist|        | |
|  |    filesystem]        |  |    Title|Date]          | |
|  |                       |  |                         | |
|  |                       |  |   - Song 1              | |
|  |                       |  |   - Song 2              | |
|  |                       |  |   - ...                 | |
|  +------------------------+  +-------------------------+ |
+----------------------------------------------------------+
|  [Controls Bar]                                          |
|  <<  |  >/||  |  >>   |   Song Title   |   00:00/03:45  |
+----------------------------------------------------------+
```

### Visual Design
- **Color Scheme**:
  - Background: #2b2b2b (dark gray)
  - Panels: #3c3c3c (lighter gray)
  - Accent: #1db954 (Spotify green for play button)
  - Text: #ffffff (white)
  - Secondary text: #b3b3b3 (light gray)
- **Font**: System default (Segoe UI on Windows, SF Pro on macOS, Ubuntu on Linux)
- **Spacing**: 8px padding between elements
- **Minimum Window Size**: 900x600 pixels

### Components

#### File Browser (Left Panel - 40% width)
- Tree view showing directory structure
- Only shows directories and audio files (.mp3, .wav)
- Double-click to enter directory
- Single-click on file to select
- "Add to Playlist" button below

#### Playlist (Right Panel - 60% width)
- Table with columns: #, Title, Artist, Duration
- Sortable by clicking column headers
- Right-click context menu: Remove, Move Up, Move Down
- Add/Remove buttons at bottom
- Save/Load playlist buttons

#### Playbar (Bottom - 60px height)
- Previous track button (<<)
- Play/Pause button (>) - larger, accent color
- Next track button (>>)
- Current track info (title - artist)
- Progress bar (clickable to seek)
- Time display (current / total)

## Functionality Specification

### Core Features
1. **File Browser**
   - Display system folders starting from home directory
   - Filter to show only folders and audio files
   - Navigate by double-clicking folders

2. **Playlist Management**
   - Add selected files to playlist
   - Remove selected tracks from playlist
   - Reorder tracks (move up/down)
   - Save playlist to .m3u file
   - Load playlist from .m3u file

3. **Playback Controls**
   - Play/Pause toggle
   - Next track
   - Previous track
   - Seek in track (click progress bar)

4. **Sorting**
   - Sort by Artist (Interpret)
   - Sort by Title
   - Sort by Date (file modification date)

5. **Metadata Display**
   - Read ID3 tags from MP3 files
   - Display title, artist if available
   - Fall back to filename if no tags

### User Interactions
- Double-click file in browser: Add to playlist
- Double-click track in playlist: Start playing
- Click Play/Pause: Toggle playback
- Click Next/Previous: Change track
- Click progress bar: Seek to position

### Edge Cases
- Empty playlist: Disable play button
- Invalid audio file: Show error, skip to next
- End of playlist: Stop or loop (option)

## Acceptance Criteria
1. Application launches without errors
2. File browser shows home directory
3. Can navigate to any folder with audio files
4. Can add files to playlist
5. Can play/pause/skip tracks
6. Progress bar shows current position and is seekable
7. Can sort playlist by title, artist, or date
8. Can save and load playlists
9. UI is clean and responsive
10. Works with MP3 and WAV files
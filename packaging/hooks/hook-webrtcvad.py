# Overrides the contrib hook, which copies metadata for the `webrtcvad` distribution. Jarvis
# installs `webrtcvad-wheels` (prebuilt, no MSVC needed), which provides the same module
# and reads its version from its own metadata.
from PyInstaller.utils.hooks import copy_metadata

datas = copy_metadata("webrtcvad-wheels")

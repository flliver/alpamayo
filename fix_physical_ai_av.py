"""
Monkey-patch fix for physical_ai_av package bug.

The bug: get_paths_info() returns empty list, causing IndexError at [0].
Fix: Check if the list is empty before accessing [0], and handle gracefully.
"""

import physical_ai_av.utils.hf_interface as hf_interface


# Save original method
_original_download_file = hf_interface.HfRepoInterface.download_file


def _patched_download_file(self, filename: str, **kwargs):
    """Patched version that handles empty get_paths_info results."""
    if self.is_file_cached(filename):
        # File already cached, just download it
        return self.api.hf_hub_download(
            filename=filename,
            cache_dir=self.cache_dir,
            local_dir=self.local_dir,
            **self.repo_snapshot_info,
            **kwargs,
        )

    # Try to get file size for confirmation
    try:
        paths_info = self.api.get_paths_info(paths=[filename], **self.repo_snapshot_info)
        if paths_info:
            file_size = paths_info[0].size
            if not self._confirm_download(file_size):
                return None
    except (IndexError, Exception) as e:
        # If we can't get file info, just proceed with download
        print(f"Warning: Could not get file info for {filename}, proceeding anyway: {e}")

    # Download the file
    return self.api.hf_hub_download(
        filename=filename,
        cache_dir=self.cache_dir,
        local_dir=self.local_dir,
        **self.repo_snapshot_info,
        **kwargs,
    )


# Apply monkey patch
hf_interface.HfRepoInterface.download_file = _patched_download_file

print("✓ Applied physical_ai_av bug fix")

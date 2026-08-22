import { useState } from "react";
import { api } from "../api";

// YouTube breaks old yt-dlp releases every few weeks, and the bundled sidecar
// lives inside the signed, read-only app bundle — so the fix is to fetch a
// fresh binary into the app data dir. `short_ytdlp_error` (ytdlp.rs) tags the
// failures this helps with by naming the button, so wherever that phrase can
// reach the user, this button has to be reachable too.
export const STALE_HINT = "Update downloader";

interface Props {
  /** Called once a newer yt-dlp is in place — re-run whatever just failed. */
  onUpdated: () => void;
  disabled?: boolean;
}

export default function UpdateDownloaderButton({ onUpdated, disabled }: Props) {
  const [updating, setUpdating] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  const run = async () => {
    setUpdating(true);
    setFailure(null);
    try {
      await api.updateYtdlp();
      onUpdated();
    } catch (e) {
      setFailure(String(e));
    } finally {
      setUpdating(false);
    }
  };

  return (
    <span className="update-downloader">
      <button
        className="btn btn-ghost btn-sm"
        onClick={() => void run()}
        disabled={updating || disabled}
      >
        {updating ? "Updating…" : "⇩ Update downloader"}
      </button>
      {failure && <span className="update-downloader-note">{failure}</span>}
    </span>
  );
}

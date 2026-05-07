import { useEffect, type MouseEvent, type ReactNode } from "react";
import { X } from "lucide-react";

type ModalOverlayProps = {
  title: string;
  onClose: () => void;
  closeLabel: string;
  children: ReactNode;
};

export function ModalOverlay({ title, onClose, closeLabel, children }: ModalOverlayProps) {
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  function handleBackdropClick(event: MouseEvent<HTMLDivElement>) {
    if (event.target === event.currentTarget) {
      onClose();
    }
  }

  return (
    <div className="modal-overlay" role="presentation" onClick={handleBackdropClick}>
      <section className="modal-overlay__panel" role="dialog" aria-modal="true" aria-label={title}>
        <header className="modal-overlay__header">
          <strong>{title}</strong>
          <button className="modal-close-button" type="button" onClick={onClose} title={closeLabel} aria-label={closeLabel}>
            <X size={16} />
          </button>
        </header>
        <div className="modal-overlay__body">{children}</div>
      </section>
    </div>
  );
}

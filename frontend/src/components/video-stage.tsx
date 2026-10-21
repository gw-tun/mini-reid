"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

export type StageRect = {
  left: string;
  top: string;
  width: string;
  height: string;
};

const FULL_RECT: StageRect = { left: "0%", top: "0%", width: "100%", height: "100%" };

/**
 * Stage that letterboxes a video of a known aspect ratio inside a flexible box.
 * The render prop receives the exact rect of the visible picture so the video
 * element, the native controls and any detection overlay stay aligned.
 */
export function VideoStage({
  aspect,
  className,
  children,
}: {
  aspect: number;
  className?: string;
  children: (rect: StageRect) => ReactNode;
}) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [rect, setRect] = useState<StageRect>(FULL_RECT);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) {
      return;
    }
    const update = () => {
      const width = stage.clientWidth;
      const height = stage.clientHeight;
      if (!width || !height || !aspect) {
        return;
      }
      const stageAspect = width / height;
      let pictureWidth = width;
      let pictureHeight = height;
      if (stageAspect > aspect) {
        pictureHeight = height;
        pictureWidth = height * aspect;
      } else {
        pictureWidth = width;
        pictureHeight = width / aspect;
      }
      setRect({
        left: `${(((width - pictureWidth) / 2) / width) * 100}%`,
        top: `${(((height - pictureHeight) / 2) / height) * 100}%`,
        width: `${(pictureWidth / width) * 100}%`,
        height: `${(pictureHeight / height) * 100}%`,
      });
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(stage);
    return () => observer.disconnect();
  }, [aspect]);

  return (
    <div ref={stageRef} className={className}>
      {children(rect)}
    </div>
  );
}

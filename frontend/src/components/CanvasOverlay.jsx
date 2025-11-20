import React, { useRef, useEffect } from 'react';

export default function CanvasOverlay({ width, height, detections, activeIndex }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // Clear previous frame
    ctx.clearRect(0, 0, width, height);

    // Draw detections
    detections.forEach((d, i) => {
      const { x1, y1, x2, y2, label, score } = d;
      
      // LOGIC: If an item is selected, dim everyone else.
      // If nothing is selected (activeIndex === null), show everyone.
      const isActive = activeIndex === null || activeIndex === i;
      const isSelected = activeIndex === i;

      ctx.globalAlpha = isActive ? 1.0 : 0.1; // Dim non-active items significantly

      // Box styling
      ctx.strokeStyle = isSelected ? '#34d399' : '#6366f1'; // Emerald-400 if selected, else Indigo
      ctx.lineWidth = isSelected ? 4 : 3;
      
      if (isSelected) {
          // Add a glow effect for the selected item
          ctx.shadowColor = '#34d399';
          ctx.shadowBlur = 15;
      } else {
          ctx.shadowBlur = 0;
      }

      // Draw Box
      ctx.beginPath();
      ctx.rect(x1, y1, x2 - x1, y2 - y1);
      ctx.stroke();

      // Reset shadow for text
      ctx.shadowBlur = 0; 

      // Label background
      const text = `${label} ${Math.round(score * 100)}%`;
      ctx.font = isSelected ? 'bold 16px Inter, sans-serif' : 'bold 14px Inter, sans-serif';
      const textWidth = ctx.measureText(text).width;
      
      ctx.fillStyle = isSelected ? '#34d399' : '#6366f1';
      ctx.fillRect(x1, y1 - (isSelected ? 28 : 24), textWidth + 10, isSelected ? 28 : 24);

      // Label Text
      ctx.fillStyle = isSelected ? '#000000' : '#ffffff';
      ctx.fillText(text, x1 + 5, y1 - 7);
    });
  }, [detections, width, height, activeIndex]);

  return (
    <canvas 
      ref={canvasRef} 
      width={width} 
      height={height} 
      className="w-full h-full object-contain pointer-events-none transition-opacity duration-300"
    />
  );
}
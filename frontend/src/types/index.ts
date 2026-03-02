export type BoundingBox = [number, number, number, number]; // [x1, y1, x2, y2]

export interface Detection {
    class: string;
    confidence: number;
    bbox: BoundingBox;
}

export interface DetectionResponse {
    detections: Detection[];
    image_size: [number, number]; // [width, height]
}

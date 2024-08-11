import cv2
import boto3

def get_image(imgpath: str) -> bytes:
    """Reads an image file and returns its contents as bytes."""
    try:
        with open(imgpath, "rb") as img:
            return img.read()
    except FileNotFoundError:
        print(f"Error: File not found - {imgpath}")
        return None

def draw_bounding_box(image, x, y, w, h, label: str) -> None:
    """Draws a bounding box and label on the image."""
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

def object_detection(imgpath: str) -> cv2.Mat:
    """Detects objects in an image using AWS Rekognition."""
    image_contain = get_image(imgpath)
    if image_contain is None:
        return None

    image = cv2.imread(imgpath)
    width, height = image.shape[1], image.shape[0]
    print("Processing Start....")

    client = boto3.client("rekognition")
    try:
        response = client.detect_labels(Image={"Bytes": image_contain}, MaxLabels=10)
    except Exception as e:
        print(f"Error: {e}")
        return None

    print("Building the block")
    for label in response["Labels"]:
        if label["Confidence"] > 80:  # adjust the confidence threshold as needed
            label_name = label["Name"]
            instances = label["Instances"]
            for instance in instances:
                bounding_box = instance["BoundingBox"]
                x = int(bounding_box["Left"] * width)
                y = int(bounding_box["Top"] * height)
                w = int(bounding_box["Width"] * width)
                h = int(bounding_box["Height"] * height)
                draw_bounding_box(image, x, y, w, h, label_name)

    return image

image_path = "./road.jpeg"
result = object_detection(image_path)

if result is not None:
    while True:
        cv2.imshow("Result", result)
        if cv2.waitKey(1) == ord("q"):
            break
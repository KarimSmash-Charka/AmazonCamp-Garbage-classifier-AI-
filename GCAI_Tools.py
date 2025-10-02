from GCAI_Types import ImageInfo

def get_image_info(image_bytes: bytes) -> ImageInfo:
    """
        Packs the bytes received from mobile app into 
        custom ImageInfo type.
    """
    from PIL import Image
    import io

    image_data = Image.open(io.BytesIO(image_bytes))
    width, height = image_data.size
    format = image_data.mode

    return {
        "width": width,
        "height":height,
        "format": format,
        "data": image_bytes
    }
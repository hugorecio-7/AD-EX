from PIL import Image, ImageDraw

def create_center_mask(image: Image.Image) -> Image.Image:
    """
    Creates a binary mask. White = Area AI is allowed to change. Black = Area to keep.
    For your ad creatives, we mask out the middle (the product blocks) and keep the 
    top (brand/logo) and bottom (CTA/price) protected.
    """
    width, height = image.size
    mask = Image.new("L", (width, height), 0) # Black image
    draw = ImageDraw.Draw(mask)
    
    # Define the editable box (e.g., leaving 20% margin at top and 30% at bottom)
    top_margin = int(height * 0.20)
    bottom_margin = int(height * 0.70)
    
    # Draw white rectangle in the middle
    draw.rectangle([(0, top_margin), (width, bottom_margin)], fill=255)
    return mask

def composite_images(original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
    """
    The magic trick for exact text preservation.
    We paste the original image OVER the generated image, but only where the mask is black.
    """
    # Invert the mask (White becomes the text areas we want to protect)
    inverted_mask = Image.eval(mask, lambda px: 255 - px)
    
    # Paste original text/banners over the generated background
    final_image = generated.copy()
    final_image.paste(original, (0, 0), inverted_mask)
    return final_image
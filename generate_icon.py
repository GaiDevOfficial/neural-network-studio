import os
from PIL import Image, ImageDraw

def create_studio_icon():
    os.makedirs('src/assets', exist_ok=True)
    
    size = 256
    img = Image.new('RGBA', (size, size), (12, 16, 32, 255))
    draw = ImageDraw.Draw(img)
    
    # Outer dark rounded rectangle
    # Draw glowing circles and connections
    center = (size // 2, size // 2)
    
    nodes = [
        (60, 80, (0, 240, 255, 255)),
        (60, 176, (0, 240, 255, 255)),
        (128, 60, (168, 85, 247, 255)),
        (128, 128, (168, 85, 247, 255)),
        (128, 196, (168, 85, 247, 255)),
        (196, 96, (255, 0, 127, 255)),
        (196, 160, (16, 185, 129, 255))
    ]
    
    # Synapse connections
    edges = [
        (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
        (2, 5), (2, 6),
        (3, 5), (3, 6),
        (4, 5), (4, 6)
    ]
    
    # Draw edges with semi-transparent lines
    for u, v in edges:
        x1, y1, _ = nodes[u]
        x2, y2, _ = nodes[v]
        draw.line([x1, y1, x2, y2], fill=(255, 255, 255, 60), width=3)
        
    # Draw nodes
    for x, y, col in nodes:
        # Glow halo
        draw.ellipse([x - 22, y - 22, x + 22, y + 22], fill=(col[0], col[1], col[2], 50))
        # Inner core
        draw.ellipse([x - 14, y - 14, x + 14, y + 14], fill=col)
        # Center white highlight
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 255, 255, 200))
        
    # Save PNG and ICO
    png_path = 'src/assets/icon.png'
    ico_path = 'src/assets/icon.ico'
    img.save(png_path, format='PNG')
    img.save(ico_path, format='ICO', sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("Icon generated successfully.")

if __name__ == '__main__':
    create_studio_icon()

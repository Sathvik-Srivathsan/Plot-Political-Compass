import json
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# configuration
# base path for data and models
base_path = "./politicalcompassdata/"

# input and output file paths
political_coordinates_file = os.path.join(base_path, "political_ideologies_coordinates.jsonl")
# model directory
trained_model_dir = os.path.join(base_path, "distilbert_checkpoints", "final_model")

output_html_file = "user_compass_prediction.html"

def load_assets():
    # load transformer model and tokenizer
    print("loading the fine-tuned transformer model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(trained_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(trained_model_dir)
        
        # set model to evaluation mode
        model.eval()

        # load ideologies data
        ideologies_data = []
        with open(political_coordinates_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # check for valid coordinates
                if data.get("coordinates") and data["coordinates"]["x"] is not None and data["coordinates"]["y"] is not None:
                    ideologies_data.append(data)

        print(f"loaded {len(ideologies_data)} existing ideologies for comparison.")
        print("model and tokenizer loaded successfully.")
        return model, tokenizer, ideologies_data

    except filenotfounderror as e:
        print(f"error: required file or directory not found - {e.filename}. ensure program 7 completed successfully.")
        return None, None, None
    except Exception as e:
        print(f"an error occurred during asset loading: {e}")
        return None, None, None

def predict_user_coordinates(user_text, model, tokenizer):
    # predict user coordinates
    print("\npredicting user input coordinates...")
    try:
        # tokenize input
        inputs = tokenizer(
            user_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        # perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_coords = outputs.logits.cpu().numpy()[0]

        predicted_x = round(float(predicted_coords[0]), 2)
        predicted_y = round(float(predicted_coords[1]), 2)
        print(f"predicted user coordinates: (x: {predicted_x}, y: {predicted_y})")
        return predicted_x, predicted_y

    except Exception as e:
        print(f"error during coordinate prediction: {e}")
        return None, None

def generate_html_visualization(user_point, all_ideologies_data):
    # generate html file for visualization
    user_x, user_y = user_point

    # convert data to json for javascript
    all_ideologies_json = json.dumps(all_ideologies_data)
    user_point_json = json.dumps({"x": user_x, "y": user_y})

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Political Compass - Your Point & All Ideologies</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #f0f2f5;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
        }}
        .compass-container {{
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            padding: 20px;
            width: 100%;
            max-width: 800px;
            aspect-ratio: 1 / 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
        }}
        canvas {{
            border: 1px solid #e0e0e0;
            background-color: #f9f9f9;
            border-radius: 8px;
            width: 100%;
            height: 100%;
        }}
        .tooltip {{
            position: absolute;
            background-color: rgba(0, 0, 0, 0.85);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s ease-in-out;
            z-index: 100;
            white-space: pre-wrap;
            max-width: 250px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            line-height: 1.4;
        }}
        .tooltip.active {{
            opacity: 1;
        }}
        .tooltip strong {{
            color: #a7f3d0;
        }}
        .axis-label {{
            font-size: 1rem;
            font-weight: 600;
            color: #555;
            position: absolute;
        }}
        .quadrant-label {{
            font-size: 0.9rem;
            font-weight: 600;
            color: #777;
            text-align: center;
            position: absolute;
            pointer-events: none;
        }}
        .legend {{
            margin-top: 20px;
            background-color: #ffffff;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
            display: flex;
            gap: 20px;
            font-size: 0.9rem;
            color: #333;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
        }}
        .legend-color-box {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 6px;
            border: 1px solid #ccc;
        }}
    </style>
</head>
<body class="flex flex-col items-center justify-center p-4">
    <h1 class="text-2xl font-bold mb-6 text-gray-800">your political compass point & all ideologies</h1>
    <div class="compass-container relative">
        <canvas id="politicalCompassCanvas"></canvas>
        <div id="tooltip" class="tooltip"></div>

        <div class="axis-label top-4 left-1/2 -translate-x-1/2">authoritarian</div>
        <div class="axis-label bottom-4 left-1/2 -translate-x-1/2">libertarian</div>
        <div class="axis-label left-4 top-1/2 -translate-y-1/2 -rotate-90">left</div>
        <div class="axis-label right-4 top-1/2 -translate-y-1/2 rotate-90">right</div>

        <div class="quadrant-label" style="top: 15%; left: 75%; transform: translate(-50%, -50%);">authoritarian right</div>
        <div class="quadrant-label" style="top: 15%; left: 25%; transform: translate(-50%, -50%);">authoritarian left</div>
        <div class="quadrant-label" style="top: 85%; left: 25%; transform: translate(-50%, -50%);">libertarian left</div>
        <div class="quadrant-label" style="top: 85%; left: 75%; transform: translate(-50%, -50%);">libertarian right</div>
    </div>

    <div class="legend">
        <div class="legend-item">
            <div class="legend-color-box" style="background-color: #ef4444;"></div>
            <span>your predicted point</span>
        </div>
        <div class="legend-item">
            <div class="legend-color-box" style="background-color: #60a5fa;"></div>
            <span>all existing ideologies</span>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('politicalCompassCanvas');
        const ctx = canvas.getContext('2d');
        const tooltip = document.getElementById('tooltip');

        let scale = 0;
        let centerX = 0;
        let centerY = 0;

        // data embedded directly from python
        const userPredictedPoint = {user_point_json};
        const allIdeologiesData = {all_ideologies_json};

        function drawCompass() {{
            const container = canvas.parentElement;
            canvas.width = container.clientWidth - 40;
            canvas.height = container.clientHeight - 40;

            centerX = canvas.width / 2;
            centerY = canvas.height / 2;
            scale = (math.min(canvas.width, canvas.height) / 2) / 100;

            ctx.clearrect(0, 0, canvas.width, canvas.height);

            // draw background quadrants
            const quadrantcolors = {{
                'top-right': '#f8d7da',
                'top-left': '#d4edda',
                'bottom-left': '#d1ecf1',
                'bottom-right': '#fff3cd'
            }};

            ctx.globalalpha = 0.6;
            ctx.fillstyle = quadrantcolors['top-right']; ctx.fillrect(centerx, 0, centerx, centery);
            ctx.fillstyle = quadrantcolors['top-left']; ctx.fillrect(0, 0, centerx, centery);
            ctx.fillstyle = quadrantcolors['bottom-left']; ctx.fillrect(0, centery, centerx, centery);
            ctx.fillstyle = quadrantcolors['bottom-right']; ctx.fillrect(centerx, centery, centerx, centery);
            ctx.globalalpha = 1.0;

            // draw axes
            ctx.strokestyle = '#333';
            ctx.linewidth = 2;
            ctx.beginpath();
            ctx.moveto(centerx, 0); ctx.lineto(centerx, canvas.height);
            ctx.moveto(0, centery); ctx.lineto(canvas.width, centery);
            ctx.stroke();

            // draw center point
            ctx.fillstyle = '#333';
            ctx.beginpath();
            ctx.arc(centerx, centery, 5, 0, math.pi * 2);
            ctx.fill();

            // draw grid lines
            ctx.strokestyle = '#e0e0e0';
            ctx.linewidth = 0.5;
            for (let i = -100; i <= 100; i += 20) {{
                ctx.beginpath();
                ctx.moveto(centerx + i * scale, 0); ctx.lineto(centerx + i * scale, canvas.height); ctx.stroke();
                ctx.beginpath();
                ctx.moveto(0, centery - i * scale); ctx.lineto(canvas.width, centery - i * scale); ctx.stroke();
            }}

            // draw all existing ideologies
            allIdeologiesData.forEach(ideology => {{
                const px = centerx + ideology.coordinates.x * scale; 
                const py = centery - ideology.coordinates.y * scale; 

                ctx.fillstyle = '#60a5fa';
                ctx.beginpath();
                ctx.arc(px, py, 4, 0, math.pi * 2);
                ctx.fill();
            }});

            // draw user's predicted point
            const userpx = centerx + userPredictedPoint.x * scale;
            const userpy = centery - userPredictedPoint.y * scale;

            ctx.fillstyle = '#ef4444';
            ctx.beginpath();
            ctx.arc(userpx, userpy, 7, 0, math.pi * 2);
            ctx.fill();
            ctx.strokestyle = '#a00';
            ctx.linewidth = 2;
            ctx.stroke();
        }}

        function getMousePos(canvas, evt) {{
            const rect = canvas.getBoundingClientRect();
            return {{
                x: evt.clientX - rect.left,
                y: evt.clientY - rect.top
            }};
        }}

        function showTooltip(data, mouseX, mouseY) {{
            let content = '';
            // check for ideology name
            if (data.ideology_name) {{
                content = `
                    <div><strong>${{data.ideology_name}}</strong></div>
                    <div>(x: ${{data.coordinates.x}}, y: ${{data.coordinates.y}})</div>
                    <div class="mt-1">${{data.article_body.substring(0, 150)}}...</div>
                `;
            }} else {{
                content = `
                    <div><strong>your predicted point</strong></div>
                    <div>(x: ${{data.x}}, y: ${{data.y}})</div>
                `;
            }}
            tooltip.innerHTML = content;
            tooltip.style.left = `${{mouseX + 15}}px`;
            tooltip.style.top = `${{mouseY + 15}}px`;
            tooltip.classList.add('active');
        }}

        function hideTooltip() {{
            tooltip.classList.remove('active');
        }}

        canvas.addEventListener('mousemove', (e) => {{
            const mousePos = getMousePos(canvas, e);
            let hoveredPoint = null;

            // check user's point
            const userPx = centerX + userPredictedPoint.x * scale;
            const userPy = centerY - userPredictedPoint.y * scale;
            const userDistance = math.sqrt((mousePos.x - userPx)**2 + (mousePos.y - userPy)**2);
            if (userDistance < 9) {{
                hoveredPoint = userPredictedPoint;
            }}

            // check existing ideologies
            if (!hoveredPoint) {{
                for (const ideology of allIdeologiesData) {{
                    const px = centerX + ideology.coordinates.x * scale;
                    const py = centery - ideology.coordinates.y * scale;
                    const distance = math.sqrt((mousePos.x - px)**2 + (mousePos.y - py)**2);

                    if (distance < 6) {{
                        hoveredPoint = ideology;
                        break;
                    }}
                }}
            }}

            if (hoveredPoint) {{
                showTooltip(hoveredPoint, e.clientX, e.clientY);
            }} else {{
                hideTooltip();
            }}
        }});

        canvas.addEventListener('mouseout', hideTooltip);
        window.addEventListener('resize', drawCompass);

        // initial draw
        drawCompass();
    </script>
</body>
</html>
    """

    with open(output_html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"visualization saved to '{output_html_file}'. open this file in your browser.")
    import webbrowser
    webbrowser.open(output_html_file)


def main():
    model, tokenizer, ideologies_data = load_assets()

    if not all([model, tokenizer, ideologies_data]):
        print("failed to load all necessary assets. exiting.")
        return

    print("\n--- political compass predictor for user input ---")
    print("enter a paragraph about your political views. type 'end' on a new line to finish.")

    user_lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        user_lines.append(line)

    user_paragraph = "\n".join(user_lines).strip()

    if not user_paragraph:
        print("no input provided. exiting.")
        return

    user_x, user_y = predict_user_coordinates(user_paragraph, model, tokenizer)

    if user_x is None or user_y is None:
        print("could not predict coordinates for your input. exiting.")
        return

    generate_html_visualization((user_x, user_y), ideologies_data)
    print("\n--- program finished ---")

if __name__ == "__main__":
    main()

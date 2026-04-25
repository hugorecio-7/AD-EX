from ultralytics import SAM
import os

# Utilitza la teva ruta a la imatge
image_path = "data/assets/creative_501058.png"

# Carreguem el model SAM (descarregarà el fitxer la primera vegada)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

model_name = "sam2.1_t.pt"
model_path = os.path.join(project_root, "models", model_name)

print("Carregant SAM (Segment Anything)...")
model = SAM(model_path) # Model ràpid


# Executem per segmentar-ho ABSOLUTAMENT TOT
resultats = model(image_path)

# Guarda el resultat visual
output_path = "output/resultat_sam3.jpg"
resultats[0].save(filename=output_path)
print(f"Obre {output_path} per veure la màgia.")

# Un cop tens la variable "resultats" generada per SAM...
resultat = resultats[0]

print("\n--- DADES EXACTES DELS OBJECTES DETECTATS ---")

# Obtenim les mides de la imatge original per calcular percentatges
alt_img, ample_img = resultat.orig_shape
area_total = alt_img * ample_img

objectes_extrets = []

# Iterem sobre cada "caixa" que ha trobat SAM
for index, box in enumerate(resultat.boxes):
    # Coordenades [x_min, y_min, x_max, y_max]
    coords = box.xyxy[0].tolist()
    coords = [round(c, 2) for c in coords]
    
    # Calculem l'amplada i l'alçada de l'objecte
    amplada_obj = coords[2] - coords[0]
    alcada_obj = coords[3] - coords[1]
    
    # Calculem quin percentatge de la pantalla ocupa
    area_obj = amplada_obj * alcada_obj
    pct_pantalla = (area_obj / area_total) * 100
    
    # Si vas usar SAM3 amb textos, aquí tindràs el nom (ex: "button")
    # Si vas usar SAM clàssic, només posarà "item" o "object"
    classe_id = int(box.cls[0])
    etiqueta = resultat.names[classe_id] if hasattr(resultat, 'names') else "objecte"
    
    dades_objecte = {
        "id": index + 1,
        "etiqueta": etiqueta,
        "coordenades_xyxy": coords,
        "percentatge_pantalla": round(pct_pantalla, 2)
    }
    objectes_extrets.append(dades_objecte)

# Ordenem els objectes del més gran al més petit (sol ser més útil)
objectes_extrets = sorted(objectes_extrets, key=lambda x: x["percentatge_pantalla"], reverse=True)

# Imprimim només el top 5 per no inundar la terminal
for obj in objectes_extrets[:5]:
    print(f"📦 ID: {obj['id']} | Tipus: {obj['etiqueta']}")
    print(f"   Coordenades: {obj['coordenades_xyxy']}")
    print(f"   Ocupa un {obj['percentatge_pantalla']}% de l'anunci")
    print("-" * 40)
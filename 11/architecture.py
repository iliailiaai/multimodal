

input_question = "Where is the dog and what color is the cat?"
img = Image.open("dog_cat_image.jpg") 

detection_model = ""
vision_model = ""
language_embedder = ""
quantity_predictor = ""
out_projector = ""

crops = detection_model.make_crops(img, input_question) # B, Cr, Ch, H, W (Cr - crops, Ch - channels, rgb)

vis_embeddings = vision_model.embed(crops) # B, Cr, T, H 
lang_embeddings = language_embedder.embed(input_question) # B, T, H

quantity_predictions = quantity_predictor.forward(vis_embeddings, lang_embeddings) # B, Cr, Q (Q - Quantity)

out_embeddings = out_projector.forward(quantity_predictions, lang_embeddings, vis_embeddings) # B, T, H
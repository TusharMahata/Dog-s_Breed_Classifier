import tensorflow as tf
import tensorflow_hub as hub


my_model = tf.keras.models.load_model(
       ('dog_clf_final.h5'),
       custom_objects={'KerasLayer': hub.KerasLayer}
)

url = 'forest.jpg'


class_names =['Afghan_hound', 'African_hunting_dog', 'Airedale',
       'American_Staffordshire_terrier', 'Appenzeller',
       'Australian_terrier', 'Bedlington_terrier', 'Bernese_mountain_dog',
       'Blenheim_spaniel', 'Border_collie', 'Border_terrier',
       'Boston_bull', 'Bouvier_des_Flandres', 'Brabancon_griffon',
       'Brittany_spaniel', 'Cardigan', 'Chesapeake_Bay_retriever',
       'Chihuahua', 'Dandie_Dinmont', 'Doberman', 'English_foxhound',
       'English_setter', 'English_springer', 'EntleBucher', 'Eskimo_dog',
       'French_bulldog', 'German_shepherd', 'German_shorthaired_pointer',
       'Gordon_setter', 'Great_Dane', 'Great_Pyrenees',
       'Greater_Swiss_Mountain_dog', 'Ibizan_hound', 'Irish_setter',
       'Irish_terrier', 'Irish_water_spaniel', 'Irish_wolfhound',
       'Italian_greyhound', 'Japanese_spaniel', 'Kerry_blue_terrier',
       'Labrador_retriever', 'Lakeland_terrier', 'Leonberg', 'Lhasa',
       'Maltese_dog', 'Mexican_hairless', 'Newfoundland',
       'Norfolk_terrier', 'Norwegian_elkhound', 'Norwich_terrier',
       'Old_English_sheepdog', 'Pekinese', 'Pembroke', 'Pomeranian',
       'Rhodesian_ridgeback', 'Rottweiler', 'Saint_Bernard', 'Saluki',
       'Samoyed', 'Scotch_terrier', 'Scottish_deerhound',
       'Sealyham_terrier', 'Shetland_sheepdog', 'ShihTzu',
       'Siberian_husky', 'Staffordshire_bullterrier', 'Sussex_spaniel',
       'Tibetan_mastiff', 'Tibetan_terrier', 'Walker_hound', 'Weimaraner',
       'Welsh_springer_spaniel', 'West_Highland_white_terrier',
       'Yorkshire_terrier', 'affenpinscher', 'basenji', 'basset',
       'beagle', 'blackandtan_coonhound', 'bloodhound', 'bluetick',
       'borzoi', 'boxer', 'briard', 'bull_mastiff', 'cairn', 'chow',
       'clumber', 'cocker_spaniel', 'collie', 'curlycoated_retriever',
       'dhole', 'dingo', 'flatcoated_retriever', 'giant_schnauzer',
       'golden_retriever', 'groenendael', 'keeshond', 'kelpie',
       'komondor', 'kuvasz', 'malamute', 'malinois', 'miniature_pinscher',
       'miniature_poodle', 'miniature_schnauzer', 'otterhound',
       'papillon', 'pug', 'redbone', 'schipperke', 'silky_terrier',
       'softcoated_wheaten_terrier', 'standard_poodle',
       'standard_schnauzer', 'toy_poodle', 'toy_terrier', 'vizsla',
       'whippet', 'wirehaired_fox_terrier']


def model_pipeline(image):
  image = image/255.

  input_image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

  pred = my_model.predict(tf.expand_dims(input_image_tensor, axis=0))
  pred_class = class_names[pred.argmax()]
  print(pred_class)
  return pred_class


import os

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(BASE_PATH, '../data')
SCRIPTS_PATH = os.path.join(BASE_PATH, '../scripts')

LOGS_PATH = os.path.join(BASE_PATH, '../lightning_logs')

RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed')

FLICKER_TEXT_PATH = os.path.join(RAW_DATA_PATH, 'Flickr8k_text')
FLICKER_IMAGES_PATH = os.path.join(RAW_DATA_PATH, 'Flicker8k_Dataset')

RAW_CAPTIONS_PATH = os.path.join(FLICKER_TEXT_PATH, 'Flickr8k.token.txt')
RAW_LEMMATIZED_CAPTIONS_PATH = os.path.join(FLICKER_TEXT_PATH, 'Flickr8k.lemma.token.txt')

ORIGINAL_TRAIN_PATH = os.path.join(FLICKER_TEXT_PATH, 'Flickr_8k.trainImages.txt')
ORIGINAL_VALID_PATH = os.path.join(FLICKER_TEXT_PATH, 'Flickr_8k.devImages.txt')
ORIGINAL_TEST_PATH = os.path.join(FLICKER_TEXT_PATH, 'Flickr_8k.testImages.txt')

FOLDS_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'images_folds.csv')
IMAGE_EMBEDDINGS_PATH = os.path.join(PROCESSED_DATA_PATH, 'image_embeddings.pth')
IMAGE_EMBEDDINGS_MAP_PATH = os.path.join(PROCESSED_DATA_PATH, 'image_embeddings_map.csv')

PREPROCESSED_CAPTIONS_PATH = os.path.join(PROCESSED_DATA_PATH, 'captions.csv')
PREPROCESSED_LEMMATIZED_CAPTIONS_PATH = os.path.join(PROCESSED_DATA_PATH, 'captions_lemmatized.csv')

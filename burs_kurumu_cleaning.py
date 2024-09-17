import numpy as np
import pandas as pd
import re
from fuzzywuzzy import process, fuzz
import Levenshtein
from unidecode import unidecode


def clean_turkish_text(text):
    if pd.isna(text) or text == "-":
        return "Burs Yok"
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def standardize_turkish_scholarship(df, column='Burs Aldigi Baska Kurum', threshold=70):
    df[column] = df[column].apply(clean_turkish_text)
    
    unique_values = df[column].unique()
    standardized = {}
    
    # Known providers - add more as needed
    known_providers = [
        "tübitak",
        "yök",
        "tev",
        "kyk",
        "devlet",
        "odtü",
        "odtü geliştirme vakfı",
        "itü",
        "bogazici",
        "sabancı",
        "koç",
        "bilkent",
        "gsb",
        "yıldız teknik",
        "çağdaş yaşamı destekleme derneği",
        "istanbul ticaret odası",
        "mehmet zorlu vakfı",
        "istanbul büyükşehir belediyesi",
        "toplum gönüllüleri vakfı",
        "tog",
        "özyeğin üniversitesi",
        "tobb etü",
        "rönesans eğitim vakfı",
        # Add more known providers here
    ]
    
    manual_corrections = {
        'ab': 'Avrupa Birliği',
        'tübıtak': 'tübitak',
        'tubitak': 'tübitak',
        'kredi ve yurtlar kurumu': 'kyk',
        'kredi yurtlar kurumu': 'kyk',
        'kredi ve yurtlar genel müdürlüğü': 'kyk',
        'kredi ve yurtlar müdürlüğü': 'kyk',
        'kredi ve yurtlar kurumu kyk': 'kyk',
        'kyk kredi ve yurtlar kurumu': 'kyk',
        'kyk kredi ve yurtlar genel müdürlüğü': 'kyk',
        'kredi yurtlar kurumu bursu': 'kyk',
        'türk eğitim vakfı': 'tev',
        'gençlik ve spor bakanlığı': 'gsb',
        'gençlik spor bakanlığı': 'gsb',
        'gençlik ve spor bakanlığı kyk bursu': 'gsb', # choice?
        'gençlik ve spor bakanlığı kyk': 'gsb', # choice?
        'orta doğu teknik üniversitesi': 'odtü',
        'istanbul teknik üniversitesi': 'itü',
        'istanbul teknik': 'itü',
        'ito' : 'istanbul ticaret odası',
        'çydd': 'çağdaş yaşamı destekleme derneği',
        'ibb': 'istanbul büyükşehir belediyesi',
        'tgv': 'toplum gönüllüleri vakfı',
        'özyegin': 'özyeğin üniversitesi',
        # Add more manual corrections here
    }
    
    for value in unique_values:
        if value == "Burs Yok":
            standardized[value] = value
        elif value in manual_corrections:
            standardized[value] = manual_corrections[value]
        else:
            ascii_value = unidecode(value)
            ascii_providers = [unidecode(v) for v in known_providers]
            
            matches = process.extractOne(ascii_value, ascii_providers)
            if matches and matches[1] >= threshold:
                standardized[value] = [v for v in known_providers if unidecode(v) == matches[0]][0]
            else:
                standardized[value] = "Diğer Burslar"  # Other Scholarships
    
    df[f'{column}_standardized'] = df[column].map(standardized)
    df[f'{column}_categorical'] = pd.Categorical(df[f'{column}_standardized'])

    # Drop the standardized column
    df = df.drop('Burs Aldigi Baska Kurum_standardized', axis=1)

    return df


# Examine 'Diğer Burslar' category
def examine_other_scholarships(df, column='Burs Aldigi Baska Kurum_categorical', n=20):
    other_scholarships = df[df[column] == 'Diğer Burslar']['Burs Aldigi Baska Kurum']
    print(f"\nTop {n} most common entries in 'Diğer Burslar' category:")
    print(other_scholarships.value_counts().head(n))


# This block will only run if this script is executed directly
if __name__ == "__main__":
    # Read the dataframe
    df = pd.read_csv('data/2022.csv', encoding='utf-8')
    
    # Apply standardization
    df = standardize_turkish_scholarship(df)
    
    # Examine 'Diğer Burslar' category
    examine_other_scholarships(df)
    
    # Print df info
    print(df.info())
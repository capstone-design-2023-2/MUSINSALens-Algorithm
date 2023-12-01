import pickle
import os

def main():
    categories = ['long_sleeve_dress',
                  'long_sleeve_outwear',
                  'long_sleeve_top',
                  'short_sleeve_dress',
                  'short_sleeve_outwear',
                  'short_sleeve_top',
                  'shorts',
                  'skirt',
                  'sling_dress',
                  'sling',
                  'trousers',
                  'vest_dress',
                  'vest'
                  ]
    
    cache_types = ['vgg', 'color', 'gabor', 'hog']
    temp = os.path.join(os.getcwd(), 'cache')

    for cache_dir in categories:
        for cache_type in cache_types:
            pickle_path = os.path.join(temp, cache_dir + " cache", cache_type)

            try:
                with open(pickle_path, "rb") as file:
                    samples = pickle.load(file)

                for sample in samples:
                    sample['img'] = sample['img'].replace('\\', '/')
                    sample['img'] = sample['img'].replace('database/', 'images/')

                print(samples)

                with open(pickle_path, "wb") as file:
                    pickle.dump(samples, file)

            except FileNotFoundError:
                print(f"File not found: {pickle_path}")

if __name__ == "__main__":
    main()
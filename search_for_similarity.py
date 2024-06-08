import re
import requests
import numpy as np
import faiss

# Это всё внутренние зависимости, нужно будет переписывать код без них
from core.services import BaseService
from users.models import User
from ai.models import AIModelEndpoint
from ai.utils import translate_text
from stores.models import Message, MessageVector

class FaissService(BaseService):

    def __init__(self, user: User) -> None:
        # Message IDs are UUIDs > incompatible with FAISS > make mapping for {FAISS index ID: Message}
        self.index_ids_to_messages = {}
        return super().__init__(user)

    def resolve_media(self, for_message: Message, k=5):
        """ Find k most similar medias for a given message using FAISS. """
        try:
            index = self.get_medias_index(for_message)

            # Здесь должен быть текстовый запрос пользователя видеоплатформы.
            texts = re.findall(r'<([^<>|]+)>', for_message.content) or \
                    [for_message.content + for_message.previous_message.content]
            text_join = '\n'.join(texts)
            translated_text = translate_text(text=text_join)
            texts = translated_text.split('\n')

            input_vector = requests.post(
                url=AIModelEndpoint.clip_endpoint().url,
                json={
                    "texts": texts
                }
            ).json()['features']
            input_vector = np.array(input_vector).astype('float32').reshape(1, -1)
            distances, indices = index.search(input_vector, k)
            vector_ids = indices.flatten().tolist()
            messages = [
                self.index_ids_to_messages[vector_id]
                for vector_id in vector_ids
                if vector_id in self.index_ids_to_messages
            ]
            return messages
        except Exception as e:
            print(f"Error resolving media: {e}")
            return Message.objects.none()

    def get_medias_index(self, for_message: Message):
        """ Build a FAISS index from media messages' vectors. """
        medias = [m for m in for_message.medias
                  if m.media_type == for_message.media_type
                  and m.vector_obj]
        d = 1024
        nlist = len(medias)  # Number of clusters
        quantizer = faiss.IndexFlatL2(d)  # The quantizer for the IVF index
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        vectors = []
        ids = []
        self.index_ids_to_messages = {} # Reset ids mappings at each creation of FAISS index
        ids_counter = 0
        for media in medias:
            media_vector: MessageVector = media.vector_obj
            if media_vector and media_vector.vector:
                vector = media_vector.get_vector()
                if vector.size != d:
                    vector = np.resize(vector, (d,))
                vectors.append(vector)
                self.index_ids_to_messages[ids_counter] = media
                ids.append(ids_counter)
                ids_counter += 1
        if vectors:
            vectors = np.array(vectors).astype('float32')
            ids = np.array(ids).astype('int64')
            if not index.is_trained:
                index.train(vectors)
            index.add_with_ids(vectors, ids)
        else:
            print("No vectors available to add to the index.")
        return index

# TODO
# TODO
# TODO
# Зависимости: faiss, numpy, requests.
# Пока что этот код ищет только одно наиболее релевантное к запросу видео, потребуется модификация.
# TODO
# TODO
# TODO
# В данном случае при каждом запросе создаётся новый индекс конкретно для него, и это работает довольно шустро,
# на моём проекте не бывало дольше нескольких секунд. И это ещё с запросами в БД.
# Если подавать медиа для поиска сразу в виде векторов - должно быть ещё быстрее,
# т.к. операции с БД обычно самые долгие

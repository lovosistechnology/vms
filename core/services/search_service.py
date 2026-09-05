import math
import cv2
import numpy as np

from ..models import PersonProfile, VisitProfile


class PhotoSearchService:
    """Service to extract feature embeddings from uploaded images and search PersonProfile database."""

    @staticmethod
    def extract_embedding_from_image(image_bytes):
        """Extract HSV color-histogram embedding from image file bytes."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                return None
            img = cv2.resize(img, (64, 128))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [8, 6], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten().tolist()
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a, b):
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search_profiles_by_image(self, image_bytes, min_score=0.40):
        """Search database PersonProfile records matching the query image embedding."""
        query_embedding = self.extract_embedding_from_image(image_bytes)
        if not query_embedding:
            return []

        results = []
        for profile in PersonProfile.objects.all().select_related('camera'):
            if not profile.embedding:
                continue
            score = self._cosine_similarity(query_embedding, profile.embedding)
            if score >= min_score:
                recent_visits = profile.visits.order_by('-counted_time')[:5]
                results.append({
                    'profile': profile,
                    'similarity_score': round(score * 100, 1),
                    'similarity_decimal': score,
                    'recent_visits': list(recent_visits),
                })

        results.sort(key=lambda x: x['similarity_decimal'], reverse=True)
        return results


_search_service = None


def get_search_service():
    global _search_service
    if _search_service is None:
        _search_service = PhotoSearchService()
    return _search_service

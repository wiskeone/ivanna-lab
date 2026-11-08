from .models import DocumentCollection, Document
from django.db.models import Q

def rag_stats(request):
    if request.user.is_authenticated:
        collections = DocumentCollection.objects.filter(
            Q(owner_id=str(request.user.id)) | Q(is_public=True),
            is_active=True
        )
        total_collections = collections.count()
        documents = Document.objects.filter(
            collection__in=collections,
            is_active=True
        )
        total_documents = documents.count()
    else:
        total_collections = 0
        total_documents = 0
    return {
        'rag_total_collections': total_collections,
        'rag_total_documents': total_documents,
    }
from django.core.management.base import BaseCommand
from ivanna_model_manager.services.vector_store_manager import VectorStoreManager
from ivanna_model_manager.models import DocumentCollection, DocumentChunk

class Command(BaseCommand):
    help = 'Управление векторными индексами'
    
    def add_arguments(self, parser):
        parser.add_argument(
            'action',
            choices=['create', 'rebuild', 'delete', 'stats', 'list'],
            help='Действие с индексами'
        )
        parser.add_argument(
            '--collection-id',
            type=int,
            help='ID коллекции'
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Применить ко всем коллекциям'
        )
    
    def handle(self, *args, **options):
        action = options['action']
        collection_id = options['collection_id']
        all_collections = options['all']
        
        vector_manager = VectorStoreManager()
        
        if action == 'create':
            self.handle_create(vector_manager, collection_id, all_collections)
        elif action == 'rebuild':
            self.handle_rebuild(vector_manager, collection_id, all_collections)
        elif action == 'delete':
            self.handle_delete(vector_manager, collection_id, all_collections)
        elif action == 'stats':
            self.handle_stats(vector_manager, collection_id, all_collections)
        elif action == 'list':
            self.handle_list(vector_manager)
    
    def handle_create(self, vector_manager, collection_id, all_collections):
        if all_collections:
            collections = DocumentCollection.objects.filter(is_active=True)
            for collection in collections:
                self.stdout.write(f"Indexing collection {collection.id}: {collection.name}")
                count = vector_manager.index_collection(collection.id)
                self.stdout.write(f"  Indexed {count} chunks")
        elif collection_id:
            count = vector_manager.index_collection(collection_id)
            self.stdout.write(f"Indexed {count} chunks for collection {collection_id}")
        else:
            self.stdout.write("Please specify --collection-id or --all")
    
    def handle_rebuild(self, vector_manager, collection_id, all_collections):
        # Аналогично create, но с очисткой старого индекса
        pass
    
    def handle_stats(self, vector_manager, collection_id, all_collections):
        if all_collections:
            collections = DocumentCollection.objects.filter(is_active=True)
            for collection in collections:
                stats = vector_manager.get_statistics(collection.id)
                self.stdout.write(f"Collection {collection.id}: {collection.name}")
                self.stdout.write(f"  Vectors: {stats['total_vectors']}")
                self.stdout.write(f"  Dimension: {stats['dimension']}")
        elif collection_id:
            stats = vector_manager.get_statistics(collection_id)
            self.stdout.write(f"Collection {collection_id} statistics:")
            for key, value in stats.items():
                self.stdout.write(f"  {key}: {value}")
        else:
            stats = vector_manager.get_statistics()
            self.stdout.write("Global index statistics:")
            for key, value in stats.items():
                self.stdout.write(f"  {key}: {value}")
    
    def handle_list(self, vector_manager):
        self.stdout.write("Available vector stores:")
        for key in vector_manager.stores.keys():
            self.stdout.write(f"  {key}")
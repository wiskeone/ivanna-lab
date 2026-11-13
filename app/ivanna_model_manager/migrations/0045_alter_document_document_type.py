from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ivanna_model_manager', '0044_rerankermodel_documentchunk_page_number_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='document_type',
            field=models.CharField(
                choices=[
                    ('TEXT', 'Текст'), ('PDF', 'PDF'), ('DOCX', 'Word'),
                    ('HTML', 'HTML'), ('MARKDOWN', 'Markdown'),
                    ('CODE', 'Исходный код'), ('CSV', 'Таблица CSV'),
                    ('EXCEL', 'Excel'), ('RTF', 'RTF'),
                    ('PPTX', 'Презентация'), ('EPUB', 'EPUB'),
                    ('UNKNOWN', 'Неизвестный'),
                ],
                default='UNKNOWN', max_length=20,
            ),
        ),
    ]

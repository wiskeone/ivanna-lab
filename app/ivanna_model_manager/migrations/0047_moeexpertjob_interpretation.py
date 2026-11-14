from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ivanna_model_manager', '0046_alter_deleteddocumentlog_document_type'),
    ]

    operations = [
        migrations.AddField(
            model_name='moeexpertjob',
            name='interpretation',
            field=models.TextField(blank=True, default=''),
        ),
    ]

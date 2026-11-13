from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('ivanna_model_manager', '0043_moeexpertjob_goal_moeexpertjob_do_patch'),
    ]

    operations = [
        migrations.CreateModel(
            name='RerankerModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(help_text='Название модели', max_length=255, unique=True)),
                ('display_name', models.CharField(blank=True, max_length=255)),
                ('model_id', models.CharField(help_text='Идентификатор на HF (например, DiTy/cross-encoder-russian-msmarco)', max_length=500)),
                ('max_length', models.IntegerField(default=512, help_text='Максимальная длина пары запрос/документ')),
                ('is_active', models.BooleanField(default=True)),
                ('is_default', models.BooleanField(default=False)),
                ('is_downloaded', models.BooleanField(default=False)),
                ('download_progress', models.IntegerField(default=0)),
                ('download_status', models.CharField(choices=[('not_started', 'Не начата'), ('downloading', 'Загружается'), ('completed', 'Завершена'), ('failed', 'Ошибка')], default='not_started', max_length=20)),
                ('download_error', models.TextField(blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'ordering': ['-is_default', '-is_active', 'name'],
                'verbose_name': 'Модель реранкинга',
                'verbose_name_plural': 'Модели реранкинга',
            },
        ),
        migrations.AddField(
            model_name='documentchunk',
            name='page_number',
            field=models.IntegerField(blank=True, help_text='Номер страницы источника', null=True),
        ),
        migrations.AddField(
            model_name='documentcollection',
            name='reranker_model',
            field=models.ForeignKey(blank=True, help_text='Реранкер для гибридного поиска (пусто = авто по языку запроса)', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='collections', to='ivanna_model_manager.rerankermodel'),
        ),
    ]

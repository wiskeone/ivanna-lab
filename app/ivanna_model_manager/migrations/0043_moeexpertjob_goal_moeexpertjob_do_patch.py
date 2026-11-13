from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ivanna_model_manager', '0042_alter_finetuningjob_status_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='moeexpertjob',
            name='goal',
            field=models.CharField(
                choices=[
                    ('amplify', 'amplify — усилить тему (мягкий MoE)'),
                    ('ablate', 'ablate — подавить тему (WSP-абляция, как Эксп. 1)'),
                ],
                default='amplify',
                max_length=10,
                verbose_name='Цель операции',
            ),
        ),
        migrations.AddField(
            model_name='moeexpertjob',
            name='do_patch',
            field=models.BooleanField(
                default=True,
                help_text='Снять флажок = только скан и визуализация, без патча и чекпойнта',
            ),
        ),
    ]

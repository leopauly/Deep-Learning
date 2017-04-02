#$ -cwd -V
#$ -l h_rt=24:00:00
#$ -pe smp 4
#$ -l h_vmem=12G
#$ -m be
#$ -M cnlp@leeds.ac.uk
module load singularity
singularity exec /nobackup/containers/ds1.img python < vggnet.py >> vgglog.txt

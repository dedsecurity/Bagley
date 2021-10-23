printf "Instalando pacotes"

pip3 install -r requirements.txt

sudo apt install dirb -y
sudo apt install host -y
sudo apt install bin9-host -y
sudo apt install netcat
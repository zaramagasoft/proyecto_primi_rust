#![recursion_limit = "512"]
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::{Tensor, TensorData};
use burn::backend::autodiff::Autodiff; 
use serde::Deserialize;
use std::error::Error;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::module::Module;
use burn::config::Config;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};

// --- BACKEND ---
type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

// --- MODELO NÚMEROS ---
#[derive(Module, Debug)]
pub struct Model<B: burn::tensor::backend::Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    layer3: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 6)] input_size: usize,
    #[config(default = 512)] hidden_size: usize,
    #[config(default = 6)] output_size: usize,
}

impl ModelConfig {
    pub fn init<B: burn::tensor::backend::Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            layer1: LinearConfig::new(self.input_size, self.hidden_size).init(device),
            layer2: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            layer3: LinearConfig::new(self.hidden_size, self.output_size).init(device),
            activation: Relu::new(),
        }
    }
}

// --- ESTRUCTURAS DE DATOS ---
#[derive(Debug, Deserialize)]
struct Sorteo {
    n1: f32, n2: f32, n3: f32, n4: f32, n5: f32, n6: f32,
    c: f32, #[serde(rename = "Año")] anio: i32,
    #[serde(rename = "Mes")] mes: i32, #[serde(rename = "Día")] dia: i32,
}

#[derive(Debug, Deserialize)]
struct Reintegro {
    #[serde(rename = "R")] r: f32,
    #[serde(rename = "Ano")] anio: i32,
    #[serde(rename = "Mes")] mes: i32,
    #[serde(rename = "Dia")] dia: i32,
}

// --- LÓGICA DE CARGA ---
fn cargar_csv() -> Result<(Vec<Sorteo>, Vec<Reintegro>), Box<dyn Error>> {
    println!("--> Leyendo archivos CSV...");
    let mut rdr_n = csv::ReaderBuilder::new().delimiter(b',').from_path("data/datos_loteria_procesado.csv")?;
    let mut sorteos = Vec::new();
    for result in rdr_n.deserialize() { sorteos.push(result?); }

    let mut rdr_r = csv::ReaderBuilder::new().delimiter(b',').from_path("data/reintegros_convertidos.csv")?;
    let mut reintegros = Vec::new();
    for result in rdr_r.deserialize() { reintegros.push(result?); }

    Ok((sorteos, reintegros))
}

fn entrenar_y_predecir(sorteos_raw: Vec<Sorteo>, reintegros_raw: Vec<Reintegro>) {
    let device = WgpuDevice::DefaultDevice;
    
    // 1. ENTRENAMIENTO NÚMEROS
    let num_sorteos = sorteos_raw.len();
    let mut datos = Vec::with_capacity(num_sorteos * 6);
    for s in &sorteos_raw {
        datos.extend_from_slice(&[s.n1/49., s.n2/49., s.n3/49., s.n4/49., s.n5/49., s.n6/49.]);
    }

    let inputs_tensor: Tensor<MyAutodiffBackend, 2> = Tensor::from_floats(
        TensorData::new(datos, [num_sorteos, 6]), &device
    );

    let mut model = ModelConfig::new().init::<MyAutodiffBackend>(&device);
    let mut optim = AdamConfig::new().init::<MyAutodiffBackend, Model<MyAutodiffBackend>>();
    
    let inputs = inputs_tensor.clone().slice([0..num_sorteos-1, 0..6]);
    let targets = inputs_tensor.clone().slice([1..num_sorteos, 0..6]);

    println!("--> Entrenando Números (5000 épocas)...");
    for epoch in 1..=5000 {
        let x = model.layer1.forward(inputs.clone());
        let x = model.activation.forward(x);
        let x = model.layer2.forward(x);
        let x = model.activation.forward(x);
        let pred = model.layer3.forward(x);
        let loss = pred.sub(targets.clone()).powf_scalar(2.0).mean();
        
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(0.001, model, grads);
        if epoch % 500 == 0 { println!("Época [{}] Números OK", epoch); }
    }

    // 2. ENTRENAMIENTO REINTEGRO
    println!("--> Entrenando Reintegro (500 épocas)...");
    let mut datos_r = Vec::with_capacity(reintegros_raw.len());
    for r in &reintegros_raw { datos_r.push(r.r / 9.0); }
    
    let num_r = datos_r.len();
    let tensor_r: Tensor<MyAutodiffBackend, 2> = Tensor::from_floats(
        TensorData::new(datos_r, [num_r, 1]), &device
    );

    let mut model_r = LinearConfig::new(1, 16).init::<MyAutodiffBackend>(&device);
    let mut optim_r = AdamConfig::new().init::<MyAutodiffBackend, Linear<MyAutodiffBackend>>();
    
    let r_in = tensor_r.clone().slice([0..num_r-1, 0..1]);
    let r_tar = tensor_r.clone().slice([1..num_r, 0..1]);

    for _ in 1..=500 {
        let p = model_r.forward(r_in.clone());
        let l = p.sub(r_tar.clone()).powf_scalar(2.0).mean();
        let grads = l.backward();
        let grads = GradientsParams::from_grads(grads, &model_r);
        model_r = optim_r.step(0.01, model_r, grads);
    }

    // --- PREDICCIÓN FINAL ---
    let ultimo_n = inputs_tensor.slice([num_sorteos-1..num_sorteos, 0..6]);
    let pred_n = model.layer3.forward(model.activation.forward(model.layer2.forward(model.activation.forward(model.layer1.forward(ultimo_n)))));
    
    let ultimo_r = tensor_r.slice([num_r-1..num_r, 0..1]);
    let pred_r = model_r.forward(ultimo_r);

    println!("\n-------------------------------------------");
    println!("   BOLETO DE LA R9 390 PARA EL DOMINGO     ");
    println!("-------------------------------------------");
    
    if let Ok(nums) = pred_n.mul_scalar(49.0).into_data().as_slice::<f32>() {
        let mut n_vec = nums.to_vec();
        n_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        print!(" COMBINACIÓN: ");
        for n in n_vec { print!("{:.0} ", n.clamp(1.0, 49.0)); }
        println!();
    }

    if let Ok(r_val) = pred_r.mul_scalar(9.0).into_data().as_slice::<f32>() {
        println!(" REINTEGRO:   {:.0}", r_val[0].clamp(0.0, 9.0).round());
    }
    println!("-------------------------------------------");
}

fn main() {
    if let Ok((s, r)) = cargar_csv() {
        entrenar_y_predecir(s, r);
    }
}
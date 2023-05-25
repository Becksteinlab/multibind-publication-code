use rand::Rng;

// Hold system information
#[derive(Copy, Clone, Debug)]
struct System {
    g0: f64,
    g1: f64,
    g2: f64,
    k01: f64,
    k10: f64,
    k12: f64,
    k21: f64,
}

impl System {
    // Create a new thermodynamically consistent system based on 
    // free energies and rate magnitudes
    fn new(g0: f64, g1: f64, g2: f64, k10: f64, k21: f64) -> System {
        let d_g01: f64 = g1 - g0;
        let d_g12: f64 = g2 - g1;

        System {
            k10: k10,
            k01: k10 * (-d_g01).exp(),
            k21: k21,
            k12: k21 * (-d_g12).exp(),
            g0: g0,
            g1: g1,
            g2: g2,
        }
    }

    // Partition function
    fn z(&self) -> f64{
        (-self.g0).exp() + (-self.g1).exp() + (-self.g2).exp()
    }

    // State probabilities
    fn prob(&self, state: State) -> f64 {
        match state {
            State::ZERO => {self.g0 / self.z()}
            State::ONE => {self.g1 / self.z()}
            State::TWO => {self.g2 / self.z()}
        }
    }

    fn t(&self, state1: State, state2: State) -> Option<f64> {
        match (state1, state2) {
            (State::ZERO, State::TWO) => {Some((self.k10 + self.k12 + self.k01) / (self.k01 * self.k12))}
            (State::TWO, State::ZERO) => {Some((self.k12 + self.k10 + self.k21) / (self.k10 * self.k21))}
            (State::ZERO, State::ONE) => {Some(1.0 / self.k01)}
            (State::ONE, State::ZERO) => {Some((self.k21 + self.k12) / (self.k10 * self.k21))}
            (State::ONE, State::TWO)  => {Some((self.k10 + self.k01)/(self.k01 * self.k12))}
            (State::TWO, State::ONE)  => {Some(1.0 / self.k21)}
            _ => None
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
enum State {
    ZERO,
    ONE,
    TWO,
}

struct Sim {
    target: State,
    start: State,
    system: System,
}

impl Sim {
    fn run(&self, h: f64) -> f64 {
        let mut rng = rand::thread_rng();

        let mut time: f64 = 0 as f64;

        let p01: f64 = 1.0 - (-self.system.k01 * h).exp();
        let p10: f64 = 1.0 - (-self.system.k10 * h).exp();
        let p12: f64 = 1.0 - (-self.system.k12 * h).exp();
        let p21: f64 = 1.0 - (-self.system.k21 * h).exp();
        
        let mut current_state = self.start.clone();

        loop {
            if current_state == self.target {
                return time;
            }

            time += h;

            let sample = rng.gen::<f64>();

            current_state = match current_state {
                State::ZERO => {
                    if (0.0 < sample) && (sample <= p01) {State::ONE}
                    else {State::ZERO}
                }
                State::ONE => {
                    if (0.0 < sample) && (sample <= p10) {State::ZERO}
                    else if (p10 < sample)  && (sample <= (p10 + p12)) {State::TWO}
                    else {State::ONE}
                }
                State::TWO => {
                    if (0.0 < sample) && (sample <= p21) {State::ONE}
                    else {State::TWO}
                }
            };
        };
    }

    fn group_run(&self, h: f64, repeats: u64) -> f64 {
        let mut total: f64 = 0.0;
        let mut n: u64 = 0;
    
        for _ in 1..repeats {
            total += self.run(h);
            n += 1;
        };

        let mean = total / (n as f64);
        1.0 / mean
    }
}

fn main() {
    let system = System::new(0.0, 1.0, 2.0, 10.0, 5.0);

    let simulation02: Sim = Sim{start: State::ZERO, target: State::TWO, system: system};
    let simulation20: Sim = Sim{start: State::TWO, target: State::ZERO, system: system};

    let simulation01: Sim = Sim{start: State::ZERO, target: State::ONE, system: system};
    let simulation10: Sim = Sim{start: State::ONE, target: State::ZERO, system: system};

    let simulation12: Sim = Sim{start: State::ONE, target: State::TWO, system: system};
    let simulation21: Sim = Sim{start: State::TWO, target: State::ONE, system: system};

    let repeats = 10000;
    let h = 0.001;

    println!("{:?}", system);

    println!("m(0, 2): {} ({})", 1.0 / simulation02.group_run(h, repeats), system.t(State::ZERO, State::TWO).unwrap());
    println!("m(2, 0): {} ({})", 1.0 / simulation20.group_run(h, repeats), system.t(State::TWO, State::ZERO).unwrap());

    println!("m(0, 1): {} ({})", 1.0 / simulation01.group_run(h, repeats), system.t(State::ZERO, State::ONE).unwrap());
    println!("m(1, 0): {} ({})", 1.0 / simulation10.group_run(h, repeats), system.t(State::ONE, State::ZERO).unwrap());

    println!("m(1, 2): {} ({})", 1.0 / simulation12.group_run(h, repeats), system.t(State::ONE, State::TWO).unwrap());
    println!("m(2, 1): {} ({})", 1.0 / simulation21.group_run(h, repeats), system.t(State::TWO, State::ONE).unwrap());

    // CASE A
    // 0 <-> 1 <-> 2 where 1 is hidden but distringuishable
    // m(0, 2)
    

    // CASE B

    // CASE C

}

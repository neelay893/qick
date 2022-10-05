from qick import *
from socProxy import makeProxy
import matplotlib.pyplot as plt
import numpy as np
import ipdb
from qick.helpers import gauss
from Experiment import ExperimentClass
import datetime

MAX_GAIN = 30000 #is there a way to get this?

class QChipProgram(AveragerProgram):
    """
    Averager program class for generating pulses from
    sequences of QChip gates. Adds the following attributes:

    gate_schedule : list
        user provided list of gates (named according to qchip config)
        and times
    pulse_schedule : list
        list of dictionaries containing pulse timing info and attributes;
        freqs, phase, dt, etc generally converted to QICK native conventions
    readout_cfg : list
        list of dictionaries specifying readout channel configs and time delays
        for triggering acquisitions
    qchip : QChip 
        qubit calibration specifications; qubit frequencies, gate parameters, etc
    wiremap: Wiremap
        specifies qubit to firmware channel mappings
    """
    def __init__(self, soccfg, cfg, wiremap, qchip, gate_program):
        """
        Parameters
        ----------
            soccfg
            cfg : dict
                includes 'rounds', 'reps', 'soft_avgs'
            wiremap : Wiremap
            qchip : QChip
            gate_program : list
                list of dicts with 'name' key specifying gate name,
                and 't' key specifying gate time (in seconds)
        """
        self.wiremap = wiremap
        self.qchip = qchip
        self.pulse_schedule = []
        self.readout_cfg = []
        self.readouts_per_ch = {}
        self._gate_program = gate_program
        gate_list = self._resolve_gates(gate_program) #if no t provided, schedule gates
        self.gate_schedule = self._schedule_gates(gate_list)
        super().__init__(soccfg, cfg)

    def _schedule_gates(self, gate_list):
        chan_last_t = {chanind: 0 for chanind in self.wiremap.chanmapqubit.values()}
        gate_schedule = []
        for gate in gate_list:
            pulses = gate.get_pulses()
            min_pulse_t = [] 
            for pulse in pulses:
                dest_t = chan_last_t[pulse.dest]
                min_pulse_t.append(dest_t - pulse.t0) #earliest gate time based on this pulse
            gate_t = max(min_pulse_t)
            for pulse in pulses:
                chan_last_t[pulse.dest] = gate_t + pulse.t0 + pulse.twidth
            gate_schedule.append({'gate': gate, 't': gate_t})

    def _resolve_gates(self, gate_program):
        """
        convert gatedict references to objects
        """
        gate_list = []
        for gatedict in gate_program:
            if isinstance(gatedict['qubit'], str):
                gatedict['qubit'] = [gatedict['qubit']]
            gatename = ''.join(gatedict['qubit']) + gatedict['name']
            gate = self.qchip.gates[gatedict]
            if 'modi' in gatedict and gatedict['modi'] is not None:
                gate = gate.get_updated_copy(gatedict['modi'])
            gate_list.append(gate)

        return gate_list

    def initialize(self):
        for k, v in self.wiremap.chanmapqubit.items():
            chtype = k.split('.')[1]
            if  chtype == 'rdrv' or chtype == 'qdrv':
                nqz = self.wiremap.nqzones[k]
                self.declare_gen(ch=v, nqz=nqz)
            elif chtype == 'read':
                self.readouts_per_ch[v] = 0
        for gate in self.gate_schedule:
            self._add_gate(gate)
        for rpulse in self.readout_cfg:
            if self.readouts_per_ch[rpulse['ch']] == 0:
                self.declare_readout(ch=rpulse['ch'], freq=rpulse['freq'], length=rpulse['length'], gen_ch=rpulse['gen_ch'])
            self.readouts_per_ch[rpulse['ch']] += 1
        self.readouts_per_experiment = max(list(self.readouts_per_ch.values()))
        self._sort_pulse_schedule()
        self.synci(200) 
        #todo: add code for checking HW and setting LO freq if applicable

    def _add_gate(self, gatedict):
        """
        Adds pulses for specified gate to pulse library and schedule. 
        Note: 'read' pulses (ADC downconversion) are interpreted as triggering 
            the relevant channel at the specified time and acquiring for twidth
            -- there is no actual pulse
        Parameters
        ----------
            gatedict : dict
                has keys: name, qubit, t, modi (optional)
        """
        #todo: how to handle virtual z here
        #todo: test LO support
        t = gatedict['t']
        gate = gatedict['gate']

        pulses = gate.get_pulses(t)
        for i, pulse in enumerate(pulses):
            qubitid = pulse.dest.split('.')[0]
            dest_type = pulse.dest.split('.')[1]
            if dest_type == 'read':
                self._add_readout_ch(pulse, qubitid)
            else:
                ch = self.wiremap.chanmapqubit[pulse.dest]
                pulsename = '{}_{}'.format(name, i)
                samples_per_clock = self.soccfg['gens'][ch]['samps_per_clk']
                dt = 1.e-6*self.cycles2us(1)/samples_per_clock #time per sample

                #hack to account for the restriction that nsamples needs to 
                #   be multiple of samples per clock. todo: think of better way to handle this
                #ipdb.set_trace()
                #this can cause bugs with arange, and change pulse width parameters. lets pad instead
                #nsamples = int(np.ceil(pulse.twidth/dt))
                #pulse.twidth = np.round(nsamples/samples_per_clock)*samples_per_clock*dt 
                #pulse_env = pulse.get_env_samples(dt)[1] 

                ipdb.set_trace()
                pulse_env = pulse.get_env_samples(dt)[1]
                pulse_env = np.pad(pulse_env, (0, 16-len(pulse_env)%16), mode='constant')

                #if pulse drives readout channel (rdrv), need to match with corresponding ADC
                if dest_type == 'rdrv':
                    adc_ch_match = self.wiremap.chanmapqubit[qubitid + '.read']
                else:
                    adc_ch_match = None
                freq = pulse.fcarrier - self.wiremap.lofreq[pulse.dest]
                regfreq = self.freq2reg(pulse.fcarrier/1.e6, ch, adc_ch_match)
                regphase = self.deg2reg(np.degrees(pulse.pcarrier), gen_ch=ch)

                maxv = self.soccfg['gens'][ch]['maxv']*self.soccfg['gens'][ch]['maxv_scale']
                self.pulse_schedule.append({'t': self.us2cycles(1.e6*pulse.t0), 'ch': ch, #todo: match ADC freq to reg
                        'freq': regfreq, 'phase': regphase, 'name': pulsename, 
                        'gain': int(MAX_GAIN*pulse.amp), 'type': 'pulse'})
                self.add_pulse(ch, pulsename, maxv*np.real(pulse_env), maxv*np.imag(pulse_env))

    def _add_readout_ch(self, pulse, qubitid):
        ch = self.wiremap.chanmapqubit[pulse.dest]
        gen_ch = self.wiremap.chanmapqubit[qubitid + '.rdrv']
        freq_mhz = pulse.fcarrier/1.e6
        length_samples = self.us2cycles(pulse.twidth*1.e6)
        t0_samples = self.us2cycles(1.e6*pulse.t0)
        for rpulse in self.readout_cfg: 
            if rpulse['ch'] == ch:
                if freq_mhz != rpulse['freq']:
                    raise Exception('Must pre-configure readout channel to single DDS frequency')
                if length_samples != rpulse['length']:
                    raise Exception('Must pre-configure readout channel to single readout length')
        self.readout_cfg.append({'t': t0_samples, 'ch': ch, 'freq': freq_mhz, 
                    'length': length_samples, 'gen_ch': gen_ch, 'type': 'read'})
        #note: this only extracts freq, ch and time info from readout pulse. need to figure out
        # support for readout envelope or phase. also, can you trigger the same channel multiple times
        # during a program?

    def _sort_pulse_schedule(self):
        self.prog_schedule = self.pulse_schedule + self.readout_cfg
        self.prog_schedule = sorted(self.prog_schedule, key=lambda pulse: pulse['t'])

    def body(self):
        for inst in self.prog_schedule:
            if inst['type'] == 'read':
                self.trigger(adcs=[inst['ch']], adc_trig_offset=inst['t'])
            elif inst['type'] == 'pulse': 
                self.set_pulse_registers(ch=inst['ch'], style='arb', freq=inst['freq'], 
                            phase=inst['phase'], gain=inst['gain'], waveform=inst['name'])
                self.pulse(ch=inst['ch'], t=inst['t'])
            else:
                raise Exception('{} type not supported'.format(inst['type']))

        self.wait_all()
        self.sync_all(200) #seems like a sensible default...

    def acquire(self, soc, **kwargs):
        """
        Calls AveragerProgram.acquire, but using the internally determined readouts_per_experiment. 
        All other args/behavior remain the same.
        """
        return AveragerProgram.acquire(self, soc, readouts_per_experiment=self.readouts_per_experiment, **kwargs)

#Todo: make experiment class here
#class Loopback(ExperimentClass):
#    """
#    Loopback Experiment basic
#    """
#
#    def __init__(self, soc=None, soccfg=None, path='', prefix='data', cfg=None, config_file=None, progress=None):
#        super().__init__(soc=soc, soccfg=soccfg, path=path, prefix=prefix, cfg=cfg, config_file=config_file, progress=progress)
#
#    def acquire(self, progress=False, debug=False):
#        prog = LoopbackProgram(self.soccfg, self.cfg)
#        self.soc.reset_gens()  # clear any DC or periodic values on generators
#        iq_list = prog.acquire_decimated(self.soc, load_pulses=True, progress=False, debug=False)
#        data={'config': self.cfg, 'data': {'iq_list': iq_list}}
#        self.data=data
#        return data
#
#    def display(self, data=None, fit=True, **kwargs):
#        if data is None:
#            data = self.data
#        plt.figure(1)
#        for ii, iq in enumerate(data['data']['iq_list']):
#            plt.plot(iq[0], label="I value, ADC %d" % (data['config']['ro_chs'][ii]))
#            plt.plot(iq[1], label="Q value, ADC %d" % (data['config']['ro_chs'][ii]))
#            plt.plot(np.abs(iq[0] + 1j * iq[1]), label="mag, ADC %d" % (data['config']['ro_chs'][ii]))
#        plt.ylabel("a.u.")
#        plt.xlabel("Clock ticks")
#        plt.title("Averages = " + str(data['config']["soft_avgs"]))
#        plt.legend()
#        plt.savefig(self.iname)
#
#    def save_data(self, data=None):
#        print(f'Saving {self.fname}')
#        super().save_data(data=data['data'])

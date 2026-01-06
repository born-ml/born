// Copyright 2025 Born ML Framework. All rights reserved.
// Use of this source code is governed by an Apache 2.0
// license that can be found in the LICENSE file.

package optim

import (
	"github.com/born-ml/born/internal/optim"
	"github.com/born-ml/born/nn"
	"github.com/born-ml/born/tensor"
)

// Optimizer interface defines the common interface for all optimizers.
//
// Optimizers update model parameters based on computed gradients.
// All optimizers implement Step() for parameter updates and ZeroGrad() for clearing gradients.
//
// Note: This is a type alias because the Step method signature references internal tensor types.
type Optimizer = optim.Optimizer

// Config represents the base configuration for optimizers.
type Config = optim.Config

// SGD (Stochastic Gradient Descent)

// SGD represents the SGD optimizer with optional momentum.
type SGD[B tensor.Backend] = optim.SGD[B]

// SGDConfig contains configuration for SGD optimizer.
type SGDConfig = optim.SGDConfig

// NewSGD creates a new SGD optimizer.
//
// Example:
//
//	backend := cpu.New()
//	model := nn.NewLinear(784, 10, backend)
//	optimizer := optim.NewSGD(
//	    model.Parameters(),
//	    optim.SGDConfig{
//	        LR:       0.01,
//	        Momentum: 0.9,
//	    },
//	    backend,
//	)
func NewSGD[B tensor.Backend](params []*nn.Parameter[B], config SGDConfig, backend B) *SGD[B] {
	return optim.NewSGD(params, config, backend)
}

// Adam (Adaptive Moment Estimation)

// Adam represents the Adam optimizer.
type Adam[B tensor.Backend] = optim.Adam[B]

// AdamConfig contains configuration for Adam optimizer.
type AdamConfig = optim.AdamConfig

// NewAdam creates a new Adam optimizer with bias correction.
//
// Example:
//
//	backend := cpu.New()
//	model := nn.NewLinear(784, 10, backend)
//	optimizer := optim.NewAdam(
//	    model.Parameters(),
//	    optim.AdamConfig{
//	        LR:      0.001,
//	        Betas:   [2]float32{0.9, 0.999},
//	        Epsilon: 1e-8,
//	    },
//	    backend,
//	)
func NewAdam[B tensor.Backend](params []*nn.Parameter[B], config AdamConfig, backend B) *Adam[B] {
	return optim.NewAdam(params, config, backend)
}

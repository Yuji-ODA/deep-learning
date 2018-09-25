
require 'mathlib'

module RandomDist
  include MathLib
  include Math
  extend self

  # 指数分布
  def expo_dist(mean)
    # 逆関数法より
    # CDF = F(X) = 1 - exp(-X/λ)
    # X = Finv(CDF) = -λ * ln(1 - CDF)
    if 0 < mean 
      - log(1 - rand) * mean
    else
      raise 'mean must be positive value!!'
    end  
  end
  
  def erlang_dist(mean, k)
    # 平均　mean /k　の指数分布乱数をk個発生させ、その和をとる
    # - mean / k * Σ(1->k)ln(1 - rand)
    # Σ(1->k)ln(1 - rand) = lnΠ(1->k)(1 - rand) 
    if 0 < mean 
      - mean / k * log(k.times.inject(1) {|v,_| v * (1 - rand)})
    else
      raise 'mean must be positive value!!'
    end  
  end

  # 正規分布
  def normal_dist(mean, sd)
    # box-muller法による
    # 独立に標準正規分布に従う変数x,yに対して、x^2+y^2が平均2の指数分布に従い
    # x−y平面の原点を中心とする同一円周上の分布が一様であることを利用する
    # つまり、まず平均2の指数分布に従う乱数rを生成し、
    # 次に半径r√上の円周からランダムにサンプルされた1点のx座標を返せばよい。
    sqrt(expo_dist(2)) * sin(2 * Math::PI * rand) * sd + mean
  end
  
  # 対数正規分布
  def log_normal_dist(mean, sd)
    # 確率変数の対数が正規分布に従う
    #  mean = exp(log_mean + log_sd^2 / 2)
    #  sd = mean^2 * exp(log_sd^2) - mean
    # より
    #  log_mean = ln(mean) - ln(1 + (sd / mean)^2) / 2
    #  log_sd = sqrt(ln(1 + (sd / mean)^2))
    if 0 < mean 
      exp(normal_dist(log(mean) - (f = log(1 + sd.fdiv(mean) ** 2)) / 2, sqrt(f)))
    else
      raise 'mean must be positive value!!'
    end  
  end

  # パレート分布
  def pareto_dist(mean, alpha)
    # 逆関数法
    mean * (alpha - 1.0) / alpha * (rand ** (-1.0 / alpha))
  end

  # ポアソン分布
  def poisson_dist(mean)
    # 平均meanの指数分布の間隔がΔt=1 の間に何区間入るかを計算する
    # 指数分布乱数は 
    #  tn = -ln(1- rand) / mean
    # 求める回数をkとすると
    #  1 ≦ Σ(0->k)tn
    # から
    #  Π(0->k)rand ≦ exp(-mean)
    # となる最小のkを求める
    if 0 < mean 
      thres = Math.exp(-mean)
      k, acc = 0, 1
      until (acc *= (1 - rand)) <= thres; k += 1 end
      k
    else
      raise 'mean must be positive value!!'
    end  
  end
end

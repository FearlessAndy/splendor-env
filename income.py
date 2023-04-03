from main import SplendorEnvSimple

env = SplendorEnvSimple()

obv, mask = env.reset()
print(obv)
print(mask)
while True:
    action = env.sample_action(mask)
    print(action)
    obv, reward, terminated, mask = env.step(action)
    print(obv)
    print(mask)
    if terminated:
        print("the winner is:" + str((env.agent_role + 1) % 2))
        break

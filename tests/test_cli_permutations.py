from speaktome.util.cli_permutations import CLIArgumentMatrix


def test_permutation_generation():
    matrix = CLIArgumentMatrix()
    matrix.add_option('--flag-a', [None])
    matrix.add_option('--num', [1, 2])
    combos = matrix.generate()
    assert combos == [['--flag-a', '--num', '1'], ['--flag-a', '--num', '2']]
